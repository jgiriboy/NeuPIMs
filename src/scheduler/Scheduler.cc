#include "Scheduler.h"

#include <cmath>
#include <cstddef>

#include "../tensor/NPUTensor.h"
#include "../tensor/PIMTensor.h"

Scheduler::Scheduler(SimulationConfig config, const cycle_type* core_cycle)
    : _config(config), _core_cycle(core_cycle), _cycles(0) {
    _max_batch_size = 1024;   // 256;   // config.max_batch_size;
    _max_active_reqs = 1024;  // 256;  // 70;
    _active_reqs = 0;
    _next_ch = 0;
    _ch_load_balancing = config.ch_load_balancing;

    // Model dimension init
    _nh = _config.model_n_head / _config.n_tp;
    _dk = _config.model_n_embd / _config.model_n_head;
    _effective_e = _nh * _dk;

    // Memory spec init
    _dram_channels = _config.dram_channels;
    _dram_page_size = _config.dram_page_size / _config.precision;
    _dram_banks_per_ch = _config.dram_banks_per_ch;

    // 1: Systolic Array Program
    // 2: PIM Program
    _model_program1 = nullptr;
    _model_program2 = nullptr;
    #ifdef TRI
    _model_program3 = nullptr;
    #endif

    _init_stage = Stage::A;
    // _init_stage = Stage::C;
    _stage = _init_stage;
    _just_one_stage = false;

    _has_stage_changed = false;

    _partition_alg_simple = true;

    // Request queue for channel
    for (int i = 0; i < _dram_channels; i++) {
        auto req_q = std::vector<Ptr<InferRequest>>();
        auto req_latency_q = std::vector<uint32_t>();
        req_q.reserve(_max_batch_size);
        req_latency_q.reserve(_max_batch_size);
        _active_request_queues.push_back(req_q);
        _active_request_latency_queues.push_back(req_latency_q);
        _active_request_accum_latencys.push_back(0);
    }

    // KV allocate by pim tile
    int model_weight = _config.model_params_b * _config.precision / _config.n_tp;  // GB
    int memory_capacity = _dram_channels;                                          // GB
    int available_for_kv = memory_capacity - model_weight;                         // GB
    int pim_tile_size = _config.dram_page_size * _dram_banks_per_ch;               // B
    _total_tiles = floor((double)available_for_kv GB / pim_tile_size);
    _total_available_tiles = _total_tiles;

    int tiles_per_channel = floor((double)_total_tiles / _dram_channels);
    _available_tiles.reserve(_dram_channels);
    for (int i = 0; i < _dram_channels; i++) {
        _available_tiles.push_back(tiles_per_channel);
    }

    spdlog::info("Total PIM tiles: {}", _total_tiles);
    spdlog::info("Tiles per channel: {}", tiles_per_channel);

    // how often to create a page per number of tokens.
    _key_period = _dram_banks_per_ch;
    _value_period = _dram_page_size;

    // how many PIM tiles compose a page.
    _key_page_size = ceil((double)_effective_e / _value_period);
    _value_page_size = ceil((double)_effective_e / _key_period);

    spdlog::info("_key_period: {}", _key_period);
    spdlog::info("_key_page_size: {}", _key_page_size);
    spdlog::info("_value_period: {}", _value_period);
    spdlog::info("_value_page_size: {}", _value_page_size);
    spdlog::info("Effective E(_nh * _dk):{}", _nh * _dk);

    // PIM GEMV latency
    _gwrite_latency = 100;
    _gemv_latency = 184;
}

void Scheduler::launch(Ptr<Model> model) {
    _model = model;
    spdlog::info("MODEL {} Launched in Scheduler", model->get_name());
}

/* Deprecated: allocate channel when making dataset */
// if return -1, it means there is no available tile for this request
int Scheduler::allocate_pim_tile(uint32_t seq_len) {
    // granularity of key
    // granularity of value

    int key_pages = ceil((double)seq_len / _key_period);
    int value_pages = ceil((double)seq_len / _value_period);

    int key_tiles = key_pages * _key_page_size;
    int value_tiles = value_pages * _value_page_size;
    int required_tiles_for_kv_cache = key_tiles + value_tiles;

    uint32_t ch;

    if (_ch_load_balancing) {
        // >> neupims: channel load balancing
        // greedy algorithm
        ch = -1;
        int min_latency = 2147483647;
        for (int i = 0; i < _dram_channels; i++) {
            int available_tiles = _available_tiles[i];
            int accum_latency = _active_request_accum_latencys[i];
            if (available_tiles < required_tiles_for_kv_cache) continue;

            // Find most lazy channel (min total latency)
            if (min_latency > accum_latency) {
                min_latency = accum_latency;
                ch = i;
            }
        }
        if (ch == -1) {
            spdlog::info("No available tiles for this request");
            return -1;
        } else {
            // push to channel that is min (sum of requests latency) among channels
            _available_tiles[ch] -= required_tiles_for_kv_cache;
        }

    } else {
        // >> newton: round-robin channel allocate
        ch = _next_ch % _dram_channels;
        int available_tiles = _available_tiles[ch];
        if (available_tiles >= required_tiles_for_kv_cache) {
            _available_tiles[ch] -= required_tiles_for_kv_cache;
        } else {
            int trial = _dram_channels;
            while (trial--) {
                ch = _next_ch % _dram_channels;
                available_tiles = _available_tiles[ch];
                if (available_tiles >= required_tiles_for_kv_cache) {
                    _available_tiles[ch] -= required_tiles_for_kv_cache;
                    break;
                }
                ch = -1;
                _next_ch++;
            }
        }
        if (ch == -1) {
            spdlog::info("No available tiles for this request");
            return -1;
        }
        _next_ch++;
    }

    assert(ch != -1);
    // spdlog::info("seqlen: {}", seq_len);
    // spdlog::info("required Key pages: {}", key_pages);
    // spdlog::info("required Value pages: {}", value_pages);
    // spdlog::info("required KV tiles: {}", required_tiles_for_kv_cache);

    _total_available_tiles -= required_tiles_for_kv_cache;
    // spdlog::info("Remain tiles: {}", _total_available_tiles);
    // spdlog::info("Remain tiles in ch#{}: {}", ch, _available_tiles[ch]);
    // spdlog::info("--------------------");
    return ch;
}

void Scheduler::allocate_requests() {
    uint32_t batch_size = 0;

    // if (_ch_load_balancing) {
    //     // sort request_queue by sequence length
    //     // for channel load balancing algorithm
    //     // greedy algorithm
    //     std::sort(_request_queue.begin(), _request_queue.end(),
    //               [this](const Ptr<InferRequest>& a, const Ptr<InferRequest>& b) {
    //                   return compare_by_seqlen(a, b);
    //               });
    // }
    for (auto it = _request_queue.begin(); it != _request_queue.end(); it++) {
        if (batch_size == _max_batch_size) break;
        Ptr<InferRequest> request = *it;
        assert(request->output_size > request->generated);

        if (!request->is_initiated) {
            int ch = request->channel;
            assert(ch < _dram_channels);
            spdlog::info("request#{} seq_len:{} channel:{}", request->id, request->input_size,
                         request->channel);
            // allocate_pim_tile(request->input_size);
            if (ch == -1) continue;

            uint32_t seq_len = request->input_size;

            std::vector<uint32_t> dim_key{_nh, _dk, seq_len};
            std::vector<uint32_t> dim_value{_nh, seq_len, _dk};

            if (_active_reqs >= _max_active_reqs) continue;
            _active_reqs++;
            // spdlog::info("Scheduler allocate request#{}(seq_len:{}) to channel {}<<",
            //              request->id, seq_len, ch);
            auto k = std::make_shared<PIMTensor>(
                name_gen(std::to_string(request->id), "KEY", std::to_string(0)), ch, dim_key,
                PIMTensorKVType::KEY, true);
            auto v = std::make_shared<PIMTensor>(
                name_gen(std::to_string(request->id), "VALUE", std::to_string(0)), ch, dim_value,
                PIMTensorKVType::VALUE, true);
            request->K_cache.push_back(k);
            request->V_cache.push_back(v);

            _active_request_queues[ch].push_back(request);
            uint32_t mha_latency = estimate_mha_latency(request);
            _active_request_latency_queues[ch].push_back(mha_latency);
            // todo: when return req, decrease accum latency
            _active_request_accum_latencys[ch] += mha_latency;

            request->is_initiated = true;
        }

        batch_size++;
    }

    // >>> load balancing check
    // spdlog::info("---------");
    // int min_latency = 9000000;
    // int max_latency = 0;
    // for (int ch = 0; ch < _dram_channels; ch++) {
    //     int channel_total_latency = _active_request_accum_latencys[ch];
    //     spdlog::info("channel #{} remain_tiles: {}, total MHA latency: {}", ch,
    //                  _available_tiles[ch], channel_total_latency);
    //     min_latency = MIN(min_latency, channel_total_latency);
    //     max_latency = MAX(max_latency, channel_total_latency);
    // }
    // spdlog::info("---------");
    // spdlog::info("MIN: {}, MAX: {}, difference: {}", min_latency, max_latency,
    //              max_latency - min_latency);
    // <<<

    // exit(-1);
}

void Scheduler::make_program() {
    std::shared_ptr<BatchedRequest> sub_batch_on_sa;
    std::shared_ptr<BatchedRequest> sub_batch_on_pim;
    #ifdef TRI
    std::shared_ptr<BatchedRequest> sub_batch_on_sa_2;
    #endif
    #ifdef TRI
    switch(_stage)
    {
        case Stage::A:
            sub_batch_on_sa = std::make_shared<BatchedRequest>(_breq2);   // exception
            sub_batch_on_sa_2 = std::make_shared<BatchedRequest>(_breq1);
            sub_batch_on_pim = std::make_shared<BatchedRequest>(_breq3);  // exception
            break;
        case Stage::F:
            sub_batch_on_sa = std::make_shared<BatchedRequest>(_breq1);
            sub_batch_on_sa_2 = std::make_shared<BatchedRequest>(_breq2);
            sub_batch_on_pim = std::make_shared<BatchedRequest>(_breq3);
            break;
        case Stage::G:
            sub_batch_on_sa = std::make_shared<BatchedRequest>(_breq2);
            sub_batch_on_sa_2 = std::make_shared<BatchedRequest>(_breq1);
            sub_batch_on_pim = std::make_shared<BatchedRequest>(_breq3);
            break;
        case Stage::H:
            sub_batch_on_sa = std::make_shared<BatchedRequest>(_breq2);
            sub_batch_on_sa_2 = std::make_shared<BatchedRequest>(_breq3);
            sub_batch_on_pim = std::make_shared<BatchedRequest>(_breq1);
            break;
        case Stage::I:
            sub_batch_on_sa = std::make_shared<BatchedRequest>(_breq3);
            sub_batch_on_sa_2 = std::make_shared<BatchedRequest>(_breq2);
            sub_batch_on_pim = std::make_shared<BatchedRequest>(_breq1);
            break;
        case Stage::J:
            sub_batch_on_sa = std::make_shared<BatchedRequest>(_breq3);
            sub_batch_on_sa_2 = std::make_shared<BatchedRequest>(_breq1);
            sub_batch_on_pim = std::make_shared<BatchedRequest>(_breq2);
            break;
        case Stage::K:
            sub_batch_on_sa = std::make_shared<BatchedRequest>(_breq1);
            sub_batch_on_sa_2 = std::make_shared<BatchedRequest>(_breq3);
            sub_batch_on_pim = std::make_shared<BatchedRequest>(_breq2);
            break;
        default:
            assert(0 && "Invalid stage");
            break;
    }

    spdlog::info("New Program for SA1 (sub-batch.size: {})", sub_batch_on_sa->_reqs.size());
    spdlog::info("New Program for SA2 (sub-batch.size: {})", sub_batch_on_sa_2->_reqs.size());
    spdlog::info("New Program for PIM (sub-batch.size: {})", sub_batch_on_pim->_reqs.size());

    _model_program1 =
        std::make_unique<StageProgram>(_model, sub_batch_on_sa, StagePlatform::SA1, _stage);
    _model_program2 =
        std::make_unique<StageProgram>(_model, sub_batch_on_pim, StagePlatform::PIM, _stage);
    _model_program3 =
        std::make_unique<StageProgram>(_model, sub_batch_on_sa_2, StagePlatform::SA2, _stage);

    refresh_status1();
    refresh_status2();
    refresh_status3();
    #else

    if (static_cast<int>(_stage) % 2 == 0) {
        sub_batch_on_sa = std::make_shared<BatchedRequest>(_breq1);
        sub_batch_on_pim = std::make_shared<BatchedRequest>(_breq2);
    } else {
        sub_batch_on_sa = std::make_shared<BatchedRequest>(_breq2);
        sub_batch_on_pim = std::make_shared<BatchedRequest>(_breq1);
    }

    spdlog::info("New Program for SA  (sub-batch.size: {})", sub_batch_on_sa->_reqs.size());
    spdlog::info("New Program for PIM (sub-batch.size: {})", sub_batch_on_pim->_reqs.size());

    // [E] These part prints yellow color in the terminal
    _model_program1 =
        std::make_unique<StageProgram>(_model, sub_batch_on_sa, StagePlatform::SA, _stage);
    _model_program2 =
        std::make_unique<StageProgram>(_model, sub_batch_on_pim, StagePlatform::PIM, _stage);


    refresh_status1();
    refresh_status2();
    
    #endif
}

int Scheduler::estimate_mha_latency(Ptr<InferRequest> request) {
    // calculate MHA latency with sequence length
    int latency = 0;
    int seq_len = request->input_size;

    // key * query
    int chunks = ceil((double)_effective_e / _dram_page_size);
    int tiles = ceil((double)seq_len / _dram_banks_per_ch);
    latency += chunks * _gwrite_latency;
    latency += chunks * tiles * _gemv_latency;

    // logit * value
    chunks = ceil((double)seq_len / _dram_page_size) * _nh;
    tiles = ceil((double)_dk / _dram_banks_per_ch);
    latency += chunks * _gwrite_latency;
    latency += chunks * tiles * _gemv_latency;

    return latency;
}

void Scheduler::group_sub_batches() {
    // if (!_config.sub_batch_mode) {
    //     //>>>
    //     // Consolidate to one batch
    //     for (int ch = 0; ch < _dram_channels; ch++) {
    //         auto req_queue = _active_request_queues[ch];
    //         for (auto it = req_queue.begin(); it != req_queue.end(); it++) {
    //             Ptr<InferRequest> request = *it;
    //             _breq1.push_back(request);
    //         }
    //     }
    //     return;
    //     //<<<
    // }
    assert(_config.sub_batch_mode);
    assert(_partition_alg_simple);

    bool ceil_turn = true;
    char bitmask = 0b00000000;
    for (int ch = 0; ch < _dram_channels; ch++) {
        auto req_queue = _active_request_queues[ch];
        auto latency_queue = _active_request_latency_queues[ch];
        assert(req_queue.size() == latency_queue.size());
        
        if (_partition_alg_simple) {

            #ifdef TRI

            std::size_t sb1_size = req_queue.size() / 3;
            std::size_t sb2_size = req_queue.size() / 3 * 2;
            std::size_t remainder = req_queue.size() % 3;

            if (req_queue.size() % 3 != 0) {
                switch(bitmask) {
                    case 0b00000000:
                        sb1_size = (req_queue.size() / 3) + 1;
                        sb2_size = (remainder==1) ? sb1_size + (req_queue.size() / 3) : sb1_size + (req_queue.size() / 3) + 1;
                        bitmask  = (remainder==1) ? 0b00000001 : 0b00000011;
                        break;   
                    case 0b00000001:
                        sb1_size = req_queue.size() / 3;
                        sb2_size = sb1_size + (req_queue.size() / 3) + 1;
                        bitmask  = (remainder==1) ? 0b00000011 : 0b00000000;
                        break;
                    case 0b00000010:
                        sb1_size = (req_queue.size() / 3) + 1;
                        sb2_size = sb1_size + (req_queue.size() / 3);
                        bitmask  = (remainder==1) ? 0b00000011 : 0b00000000;
                        break;
                    case 0b00000100:
                        sb1_size = (req_queue.size() / 3) + 1;
                        sb2_size = (remainder==1) ? sb1_size + (req_queue.size() / 3) : sb1_size + (req_queue.size() / 3) + 1;
                        bitmask  = (remainder==1) ? 0b00000101 : 0b00000000;
                        break;
                    case 0b00000011:
                        sb1_size = (remainder==1) ? req_queue.size() / 3 : (req_queue.size() / 3) + 1;
                        sb2_size = sb1_size + (req_queue.size() / 3);
                        bitmask  = (remainder==1) ? 0b00000000 : 0b00000001;
                        break;
                    case 0b00000101:
                        sb1_size = (remainder==1) ? req_queue.size() / 3 : (req_queue.size() / 3) + 1;
                        sb2_size = sb1_size + (req_queue.size() / 3) + 1;
                        bitmask  = (remainder==1) ? 0b00000000 : 0b00000001;
                        break;
                    case 0b00000110:
                        sb1_size = (remainder==1) ? (req_queue.size() / 3) + 1 : (req_queue.size() / 3) + 2;
                        sb2_size = sb1_size + (req_queue.size() / 3);
                        bitmask  = (remainder==1) ? 0b00000000 : 0b00000001;
                        break;
                    default:
                        assert(0 && "Invalid bitmask");
                        break;
                }
            }

            for (int i = 0; i < req_queue.size(); i++) {
                Ptr<InferRequest> request = req_queue[i];
                if (i < sb1_size)
                    _breq1.push_back(request);
                else if(i < sb2_size)
                    _breq2.push_back(request);
                else
                    _breq3.push_back(request);
            }

            #else

            std::size_t sb1_size = req_queue.size() / 2;

            if (req_queue.size() % 2 != 0) {
                if (ceil_turn)
                    sb1_size = ceil((double)req_queue.size() / 2);
                else
                    sb1_size = floor((double)req_queue.size() / 2);

                ceil_turn = !ceil_turn;
            }

            for (int i = 0; i < req_queue.size(); i++) {
                Ptr<InferRequest> request = req_queue[i];
                if (i < sb1_size)
                    _breq1.push_back(request);
                else
                    _breq2.push_back(request);
            }

            #endif

        // } else {
        //     auto index_lists = partition_lists_dp(latency_queue);
        //     std::vector<int> list1 = index_lists.first;
        //     std::vector<int> list2 = index_lists.second;

        //     int sum_list1_latencies = 0;
        //     int sum_list2_latencies = 0;
        //     std::string list1_str = "";
        //     std::string list2_str = "";
        //     std::string time1_str = "";
        //     std::string time2_str = "";
        //     for (auto it = list1.begin(); it != list1.end(); it++) {
        //         int req_id = *it;
        //         Ptr<InferRequest> request = req_queue[req_id];
        //         sum_list1_latencies += latency_queue[req_id];
        //         _breq1.push_back(request);
        //         list1_str += std::to_string(req_id) + ", ";
        //         time1_str += std::to_string(latency_queue[req_id]) + ", ";
        //     }
        //     for (auto it = list2.begin(); it != list2.end(); it++) {
        //         int req_id = *it;
        //         Ptr<InferRequest> request = req_queue[req_id];
        //         sum_list2_latencies += latency_queue[req_id];
        //         _breq2.push_back(request);
        //         list2_str += std::to_string(req_id) + ", ";
        //         time2_str += std::to_string(latency_queue[req_id]) + ", ";
        //     }
        //     // spdlog::info("====Channel {}====", ch);
        //     // spdlog::info("#1 sum:{:2d}, idx:[{}], time:[{}]", sum_list1_latencies, list1_str,
        //     //              time1_str);
        //     // spdlog::info("#2 sum:{:2d}, idx:[{}], time:[{}]", sum_list2_latencies, list2_str,
        //     //              time2_str);
        //     // spdlog::info("req_q.size:{}, latency_q.size:{}", req_queue.size(),
        //     // latency_queue.size());
        }
    }
    #ifdef TRI
    spdlog::info("total batch_size: {}", _breq1.size() + _breq2.size() + _breq3.size());
    #else
    spdlog::info("total batch_size: {}", _breq1.size() + _breq2.size());
    #endif
}

// Called exactly once
void Scheduler::init_batches() {
    allocate_requests();
    group_sub_batches();
}

void Scheduler::cycle() {
    #ifdef TRI
    bool step_next_stage = _model_program1 == nullptr && _model_program2 == nullptr && _model_program3 == nullptr;
    #else
    bool step_next_stage = _model_program1 == nullptr && _model_program2 == nullptr;
    #endif

    if (step_next_stage && _stage == _init_stage && !_request_queue.empty()) {
        init_batches();
        // exit(-1);
    }

    _cycles++;
    assert(_config.sub_batch_mode);
    if (_config.sub_batch_mode) {
        bool lets_make_program1 = _model_program1 == nullptr && _breq1.size() > 0;
        bool lets_make_program2 = _model_program2 == nullptr && _breq2.size() > 0;
        #ifdef TRI
        bool lets_make_program3 = _model_program3 == nullptr && _breq3.size() > 0;
        #endif

        #ifdef TRI
        if (lets_make_program1 && lets_make_program2 && lets_make_program3) {
            if (_stage == Stage::Finish) {
                cleanup_sub_batch(_breq1);
                cleanup_sub_batch(_breq2);
                cleanup_sub_batch(_breq3);
                _breq1.clear();
                _breq2.clear();
                _breq3.clear();
                return;
            } else {
                std::string red = "\033[1;31m";
                std::string reset = "\033[0m";
                spdlog::info("{}----------Stage {}----------{}", red, stageToString(_stage), reset);
                make_program();
            }
        }
        #else
        if (lets_make_program1 && lets_make_program2) {
            if (_stage == Stage::Finish) {
                cleanup_sub_batch(_breq1);
                cleanup_sub_batch(_breq2);
                _breq1.clear();
                _breq2.clear();
                return;
            } else {
                std::string red = "\033[1;31m";
                std::string reset = "\033[0m";
                spdlog::info("{}----------Stage {}----------{}", red, stageToString(_stage), reset);
                make_program();
            }
        }
        #endif
    } 
    // else {
    //     bool both_program_none = _model_program1 == nullptr && _model_program2 == nullptr;
    //     bool exist_request = _breq2.size() > 0 || _breq1.size() > 0;
    //     if (both_program_none && exist_request) {
    //         if (_stage == Stage::Finish) {
    //             cleanup_sub_batch(_breq1);
    //             cleanup_sub_batch(_breq2);
    //             _breq1.clear();
    //             _breq2.clear();
    //             return;
    //         } else {
    //             std::string red = "\033[1;31m";
    //             std::string reset = "\033[0m";
    //             spdlog::info("{}----------Stage {}----------{}", red, stageToString(_stage), reset);
    //             make_program();
    //         }
    //     }
    // }
}

void Scheduler::add_request(std::shared_ptr<InferRequest> request) {
    _request_queue.push_back(request);
}

bool Scheduler::has_completed_request() { return !_completed_request_queue.empty(); }

std::shared_ptr<InferRequest> Scheduler::pop_completed_request() {
    // spdlog::info("Scheduler::pop_completed_request()");
    auto completed_req = _completed_request_queue.front();
    _completed_request_queue.pop();
    return completed_req;
}

Tile& Scheduler::top_tile1(uint32_t core_id) {
    static Tile empty_tile = Tile{.status = Tile::Status::EMPTY};
    if (_executable_tile_queue1.empty()) {
        return empty_tile;
    } else {
        Tile& tile = _executable_tile_queue1.front();
        if (tile.status == Tile::Status::BAR) {
            return empty_tile;
        } else {
            #ifdef TRI
            tile.stage_platform = StagePlatform::SA1;
            #else
            tile.stage_platform = StagePlatform::SA;
            #endif
            return tile;
        }
    }
}

Tile& Scheduler::top_tile2(uint32_t core_id) {
    static Tile empty_tile = Tile{.status = Tile::Status::EMPTY};
    if (_executable_tile_queue2.empty()) {
        return empty_tile;
    } else {
        Tile& tile = _executable_tile_queue2.front();
        if (tile.status == Tile::Status::BAR) {
            return empty_tile;
        } else {
            tile.stage_platform = StagePlatform::PIM;
            return tile;
        }
    }
}

#ifdef TRI
Tile& Scheduler::top_tile3(uint32_t core_id) {
    static Tile empty_tile = Tile{.status = Tile::Status::EMPTY};
    if (_executable_tile_queue3.empty()) {
        return empty_tile;
    } else {
        Tile& tile = _executable_tile_queue3.front();
        if (tile.status == Tile::Status::BAR) {
            return empty_tile;
        } else {
            tile.stage_platform = StagePlatform::SA2;
            return tile;
        }
    }
}
#endif

// ??: Add base address for each addr in tiles / XXX: < necessary comment?
// ??: something wrong with functionality. seems it's not a necessary function
void Scheduler::get_tile1(uint32_t core_id) {
    if (_executable_tile_queue1.empty()) {
        return;
    } else {
        Tile& tile = _executable_tile_queue1.front();
        if (tile.status == Tile::Status::BAR) {
            RunningOperationStat stat = _finished_operation_stats[tile.operation_id];
            if (stat.launched_tiles + stat.remain_tiles == stat.total_tiles) {
                /* POP only if all lauched tiles are finished */
                _executable_tile_queue1.pop_front();
                _finished_operation_stats[tile.operation_id].launched_tiles++;
                _finished_operation_stats[tile.operation_id].remain_tiles--;
            }
            return;
        } else {
            _active_operation_stats[tile.operation_id].launched_tiles++;
            _executable_tile_queue1.pop_front();
            spdlog::debug("Operation {} Core {} Get Tile at {}", tile.optype, core_id,
                          *_core_cycle);
            return;
        }
    }
}

void Scheduler::get_tile2(uint32_t core_id) {
    if (_executable_tile_queue2.empty()) {
        return;
    } else {
        Tile& tile = _executable_tile_queue2.front();
        if (tile.status == Tile::Status::BAR) {
            RunningOperationStat stat = _finished_operation_stats[tile.operation_id];
            if (stat.launched_tiles + stat.remain_tiles == stat.total_tiles) {
                /* POP only if all lauched tiles are finished */
                _executable_tile_queue2.pop_front();
                _finished_operation_stats[tile.operation_id].launched_tiles++;
                _finished_operation_stats[tile.operation_id].remain_tiles--;
            }
            return;
        } else {
            _active_operation_stats[tile.operation_id].launched_tiles++;
            _executable_tile_queue2.pop_front();
            spdlog::debug("Operation {} Core {} Get Tile at {}", tile.optype, core_id,
                          *_core_cycle);
            return;
        }
    }
}

#ifdef TRI
void Scheduler::get_tile3(uint32_t core_id) {
    if (_executable_tile_queue3.empty()) {
        return;
    } else {
        Tile& tile = _executable_tile_queue3.front();
        if (tile.status == Tile::Status::BAR) {
            RunningOperationStat stat = _finished_operation_stats[tile.operation_id];
            if (stat.launched_tiles + stat.remain_tiles == stat.total_tiles) {
                /* POP only if all lauched tiles are finished */
                _executable_tile_queue3.pop_front();
                _finished_operation_stats[tile.operation_id].launched_tiles++;
                _finished_operation_stats[tile.operation_id].remain_tiles--;
            }
            return;
        } else {
            _active_operation_stats[tile.operation_id].launched_tiles++;
            _executable_tile_queue3.pop_front();
            spdlog::debug("Operation {} Core {} Get Tile at {}", tile.optype, core_id,
                          *_core_cycle);
            return;
        }
    }
}
#endif

//  update operation stat
//  if operation is finished
//      apply to _model_program & return true
bool Scheduler::finish_tile(uint32_t core_id, Tile& tile) {
    bool result = false;
    spdlog::debug("Tile {} Core {} Finish Tile at {}", tile.operation_id, core_id, *_core_cycle);
    assert(_active_operation_stats.find(tile.operation_id) != _active_operation_stats.end());
    assert(_finished_operation_stats.find(tile.operation_id) == _finished_operation_stats.end());
    assert(_active_operation_stats[tile.operation_id].remain_tiles > 0);
    _active_operation_stats[tile.operation_id].remain_tiles--;

    spdlog::info("Finish tile stage_platform:{}", stagePlatformToString(tile.stage_platform));
    #ifdef TRI
    
    if (tile.stage_platform == StagePlatform::SA1)
        _model_program1->finish_operation_tile(tile);
    else if (tile.stage_platform == StagePlatform::PIM)
        _model_program2->finish_operation_tile(tile);
    else{
        assert(tile.stage_platform == StagePlatform::SA2);
        _model_program3->finish_operation_tile(tile);
    }
    #else
    
    if (tile.stage_platform == StagePlatform::SA)
        _model_program1->finish_operation_tile(tile);
    else
        _model_program2->finish_operation_tile(tile);
    
    #endif
    
    if (_active_operation_stats[tile.operation_id].remain_tiles == 0) {
        result = true;
        spdlog::info("Layer {} finish at {}", _active_operation_stats[tile.operation_id].name,
                     *_core_cycle);
        spdlog::info("Total compute time {}",
                     *_core_cycle - _active_operation_stats[tile.operation_id].start_cycle);

        #ifdef TRI

        if (tile.stage_platform == StagePlatform::SA1)
            _model_program1->finish_operation(tile.operation_id);
        else if (tile.stage_platform == StagePlatform::PIM)
            _model_program2->finish_operation(tile.operation_id);
        else {
            assert(tile.stage_platform == StagePlatform::SA2);
            _model_program3->finish_operation(tile.operation_id);
        }
        #else

        if (tile.stage_platform == StagePlatform::SA)
            _model_program1->finish_operation(tile.operation_id);
        else
            _model_program2->finish_operation(tile.operation_id);

        #endif
        _finished_operation_stats[tile.operation_id] = _active_operation_stats[tile.operation_id];
        _active_operation_stats.erase(tile.operation_id);
    }


    #ifdef TRI

    if (tile.stage_platform == StagePlatform::SA1)
        refresh_status1();
    else if (tile.stage_platform == StagePlatform::PIM)
        refresh_status2();
    else{
        assert(tile.stage_platform == StagePlatform::SA2);
        refresh_status3();
    }
    #else

    if (tile.stage_platform == StagePlatform::SA)
        refresh_status1();
    else
        refresh_status2();

    #endif

    return result;
}


bool Scheduler::empty1() { return _model_program1 == nullptr; }
bool Scheduler::empty2() { return _model_program2 == nullptr; }
#ifdef TRI
bool Scheduler::empty3() { return _model_program3 == nullptr; }
#endif

bool Scheduler::running() { return !_request_queue.empty() || !_completed_request_queue.empty(); }

void Scheduler::cleanup_sub_batch(std::vector<Ptr<InferRequest>> sub_batch) {
    // < todos when the model program has finished >
    // - increment `generated` of InferRequest to 1 in batched request
    // - return completed request to client
    for (auto it = sub_batch.begin(); it != sub_batch.end(); it++) {
        Ptr<InferRequest> request = *it;

        // iteration done -> update request stat in batch
        request->is_initiated = true;
        request->generated++;

        // clear child operations of Key/Value tensor
        request->K_cache[0]->clear_child_nodes();
        request->V_cache[0]->clear_child_nodes();

        if (request->output_size == request->generated) {
            assert(request->is_initiated);
            // spdlog::info("Scheduler::return request_id: {}", request->id);
            _completed_request_queue.push(request);

            // when completed, free KV cache
            for (auto itr = _request_queue.begin(); itr != _request_queue.end();) {
                Ptr<InferRequest> cur = *itr;
                if (cur->id == request->id) {
                    itr = _request_queue.erase(itr);
                    _active_reqs--;
                    // spdlog::info("Scheduler::request {} done!", request->id);
                } else {
                    itr++;
                }
            }
        }
    }
}

void Scheduler::refresh_stage() {
    
    #ifdef TRI
    bool stage_done = _model_program1 == nullptr && _model_program2 == nullptr && _model_program3 == nullptr;
    #else
    bool stage_done = _model_program1 == nullptr && _model_program2 == nullptr;
    #endif
    
    if (stage_done) {
        std::string red = "\033[1;31m";
        std::string reset = "\033[0m";
        std::string stage_name = stageToString(_stage);
        spdlog::info("{}------- Stage {} Done -------{}", red, stage_name, reset);

        // Update stat
        _stage_stats.push_back(std::make_pair(stage_name, _cycles));

        _prev_stage = _stage;

        // Update stage
        int stageValue = static_cast<int>(_stage);
        stageValue++;
        _stage = static_cast<Stage>(stageValue);

        _has_stage_changed = true;

        assert(_config.sub_batch_mode);
        // if (!_config.sub_batch_mode) {
        //     // >> newton
        //     if (_stage == Stage::C) _stage = Stage::E;
        //     if (_stage == Stage::F) _stage = Stage::Finish;
        //     // << newton
        // }
        if (_just_one_stage) _stage = Stage::Finish;  // force to execute just one stage
    }
}


void Scheduler::finish_program1() {
    spdlog::info("Model finish at {}", *_core_cycle);
    _model_program1->log();

    _model_program1 = nullptr;
    refresh_stage();

    // cleanup_sub_batch(_breq1);
    // _breq1.clear();
}

void Scheduler::finish_program2() {
    spdlog::info("Model finish at {}", *_core_cycle);
    _model_program2->log();

    _model_program2 = nullptr;
    refresh_stage();

    // cleanup_sub_batch(_breq2);
    // _breq2.clear();
}

#ifdef TRI
void Scheduler::finish_program3() {
    spdlog::info("Model finish at {}", *_core_cycle);
    _model_program3->log();

    _model_program3 = nullptr;
    refresh_stage();

    // cleanup_sub_batch(_breq2);
    // _breq2.clear();
}
#endif

void Scheduler::refresh_status1() {
    if (_model_program1 != nullptr) {
        if (_model_program1->check_finish()) {
            finish_program1();
            // exit(0);
        }
    }
    // initiate operation
    // xxx is count_active_operations() == 0 necessary?
    if (_model_program1 != nullptr && _executable_tile_queue1.empty()) {
        // spdlog::info("executable operation count {}",
        //              _model_program1->get_executable_operations().size());
        auto op = _model_program1->get_executable_operations().front();
        spdlog::info("Start operation {}", op->get_name());
        if (count_active_operations()) {
            // for (auto& op_stat : _active_operation_stats) {
            //     spdlog::info("op stat currently in is {}", op_stat.second.name);
            // }
            if (_active_operation_stats.find(op->get_id()) != _active_operation_stats.end()) {
                return;
            }
        }

        assert(op->get_tiles().size());
        _executable_tile_queue1 = op->get_tiles();
        _active_operation_stats[op->get_id()] = RunningOperationStat{
            .id = op->get_id(),
            .name = op->get_name(),
            // xxx necessary?
            // .launched = true,
            .start_cycle = *_core_cycle,
            .total_tiles = (uint32_t)_executable_tile_queue1.size(),
            .remain_tiles = (uint32_t)_executable_tile_queue1.size(),
            .launched_tiles = 0,
        };
    } else {
        // spdlog::info("is model null {} / is executable tile queue empty {} / count active ops
        // {}",
        //              _model_program1 == nullptr, _executable_tile_queue1.empty(),
        //              count_active_operations());
        // for (auto& op_stat : _active_operation_stats) {
        //     spdlog::info("op stat currently in is {}", op_stat.second.name);
        // }
    }
}

void Scheduler::refresh_status2() {
    if (_model_program2 != nullptr) {
        if (_model_program2->check_finish()) {
            finish_program2();
            // exit(0);
        }
    }
    // initiate operation
    // xxx is count_active_operations() == 0 necessary?
    if (_model_program2 != nullptr && _executable_tile_queue2.empty()
        //  && count_active_operations() == 0) {
    ) {
        // spdlog::info("executable operation count {}",
        //              _model_program2->get_executable_operations().size());
        auto op = _model_program2->get_executable_operations().front();
        spdlog::info("Start operation {}", op->get_name());
        if (count_active_operations()) {
            if (_active_operation_stats.find(op->get_id()) != _active_operation_stats.end()) {
                return;
            }
        }

        assert(op->get_tiles().size());
        _executable_tile_queue2 = op->get_tiles();
        _active_operation_stats[op->get_id()] = RunningOperationStat{
            .id = op->get_id(),
            .name = op->get_name(),
            // xxx necessary?
            // .launched = true,
            .start_cycle = *_core_cycle,
            .total_tiles = (uint32_t)_executable_tile_queue2.size(),
            .remain_tiles = (uint32_t)_executable_tile_queue2.size(),
            .launched_tiles = 0,
        };
    }
}


#ifdef TRI
void Scheduler::refresh_status3() {
    if (_model_program3 != nullptr) {
        if (_model_program3->check_finish()) {
            finish_program3();
            // exit(0);
        }
    }
    // initiate operation
    // xxx is count_active_operations() == 0 necessary?
    if (_model_program3 != nullptr && _executable_tile_queue3.empty()
        //  && count_active_operations() == 0) {
    ) {
        // spdlog::info("executable operation count {}",
        //              _model_program2->get_executable_operations().size());
        auto op = _model_program3->get_executable_operations().front();
        spdlog::info("Start operation {}", op->get_name());
        if (count_active_operations()) {
            if (_active_operation_stats.find(op->get_id()) != _active_operation_stats.end()) {
                return;
            }
        }

        assert(op->get_tiles().size());
        _executable_tile_queue3 = op->get_tiles();
        _active_operation_stats[op->get_id()] = RunningOperationStat{
            .id = op->get_id(),
            .name = op->get_name(),
            // xxx necessary?
            // .launched = true,
            .start_cycle = *_core_cycle,
            .total_tiles = (uint32_t)_executable_tile_queue3.size(),
            .remain_tiles = (uint32_t)_executable_tile_queue3.size(),
            .launched_tiles = 0,
        };
    }
}
#endif

uint32_t Scheduler::count_active_operations() { return _active_operation_stats.size(); }

std::pair<std::vector<int>, std::vector<int>> Scheduler::partition_lists_simple(
    std::vector<uint32_t> originalVector) {
    
    assert(0 && "Is this called?");
    
    std::size_t midpointIndex = originalVector.size() / 2;

    std::vector<int> empty_vector;
    // Check if the vector is not empty
    if (originalVector.empty()) {
        std::cerr << "Vector is empty." << std::endl;
        exit(-1);
        return std::make_pair(empty_vector, empty_vector);
    }

    // Check if the midpoint index is within bounds
    if (midpointIndex >= originalVector.size()) {
        std::cerr << "Midpoint index out of bounds." << std::endl;
        exit(-1);
        return std::make_pair(empty_vector, empty_vector);  // Return an error code
    }

    // Split the vector into two vectors of the same length
    std::vector<int> list1(originalVector.begin(), originalVector.begin() + midpointIndex);
    std::vector<int> list2(originalVector.begin() + midpointIndex, originalVector.end());

    return std::make_pair(list1, list2);
}

std::pair<std::vector<int>, std::vector<int>> Scheduler::partition_lists_dp(
    std::vector<uint32_t> inputList) {
    
    assert(0 && "Is this called, too?");
    
    int totalSum = 0;
    for (int num : inputList) {
        totalSum += num;
    }

    int n = inputList.size();
    int targetSum = totalSum / 2;

    // Initialize a matrix to store intermediate results
    std::vector<std::vector<bool>> dp(n + 1, std::vector<bool>(targetSum + 1, false));

    // Base case: an empty subset can always achieve a sum of 0
    for (int i = 0; i <= n; ++i) {
        dp[i][0] = true;
    }

    // Fill the matrix using dynamic programming
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= targetSum; ++j) {
            dp[i][j] = dp[i - 1][j];
            if (j >= inputList[i - 1]) {
                dp[i][j] = dp[i][j] || dp[i - 1][j - inputList[i - 1]];
            }
        }
    }

    // Find the maximum sum that can be achieved
    int maxSum = 0;
    for (int j = targetSum; j >= 0; --j) {
        if (dp[n][j]) {
            maxSum = j;
            break;
        }
    }

    // Reconstruct the two lists
    std::vector<int> list1, list2;
    int i = n, j = maxSum;
    while (i > 0 && j > 0) {
        if (dp[i][j] && !dp[i - 1][j]) {
            list1.push_back(i - 1);
            j -= inputList[i - 1];
        } else {
            list2.push_back(i - 1);
        }
        --i;
    }

    // If there are remaining elements, add them to list1
    while (i > 0) {
        list2.push_back(i - 1);
        --i;
    }

    return std::make_pair(list1, list2);
}

void Scheduler::print_stat() {
    int prev_cycles = 0;
    for (auto stage_stat : _stage_stats) {
        auto stage_name = stage_stat.first;
        auto stage_cycles = stage_stat.second;
        auto exec_cycles = stage_cycles - prev_cycles;

        spdlog::info("Stage {} : {} cycles", stage_name, exec_cycles);

        prev_cycles = stage_cycles;
    }
}