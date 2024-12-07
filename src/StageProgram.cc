#include "StageProgram.h"

#include <cstdint>
#include <iterator>
#include <memory>
#include <vector>

#include "Common.h"
#include "Model.h"
#include "SimulationConfig.h"
#include "Stat.h"
#include "tensor/BTensor.h"
#include "tensor/NPUTensor.h"
#include "tensor/NPUTensorInner.h"
#include "tensor/PIMTensor.h"

StageProgram::StageProgram(Ptr<Model> model, Ptr<BatchedRequest> batched_request,
                           StagePlatform stage_platform, Stage stage)
    : _model(model),
      _breq(batched_request),
      _stage_platform(stage_platform),
      _stage(stage),
      _name(stagePlatformToString(stage_platform) + "_stage_" + stageToString(stage)) {
    this->init_program();
}

////////////////////////////////////////////////////////////////
// Types of stages are classified into: init, default loop, end

// (original) Sub-batch Scheduling
// #1 stands for Sub-batch 2k-1
// #2 stands for Sub-batch 2k

// |     |        init         |            default loop             |          end          |
// |     |     A    |     B    |         C        |         D        |     E     |     F     |
// |-----|:--------:|:--------:|:----------------:|:----------------:|:---------:|:---------:|
// |  SA | QKVgen#1 | QKVgen#2 | Pj/FFNs/QKVgen#1 | Pj/FFNs/QKVgen#2 | Pj/FFNs#1 | Pj/FFNs#2 |
// | PIM |     -    |  MHA#1   | MHA#2            | MHA#1            |   MHA#2   |     -     |

// (new) Three-batch Scheduling
// MHA consist of logit_softmax + attend Stage
// We divided this MHA stage into logit_softmax and attend stage
// #1 stands for Sub-batch 3k-2
// #2 stands for Sub-batch 3k-1
// #3 stands for Sub-batch 3k

// |     |                                  init (5 stages)                                         | 
// |     |     A    |        B        |        C        |          D          |          E          |
// |-----|:--------:|:---------------:|:---------------:|:-------------------:|:-------------------:|
// | SA1 | QKVgen#1 |    QKVgen#2     |    QKVgen#3     |        Pj#1         |          -          |
// | SA2 |     -    |        -        |        -        |          -          |       FFN1s#1       |
// | PIM |     -    | logit_softmax#1 |    attend#1     | logit_softmax#2     |      attend#2       |

// |     |                               default loop (6 stages)                                                     | 
// |     |        F        |        G        |        H        |       I         |        J        |         K       |
// |-----|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|
// | SA1 |      Pj#2       |    QKVgen#1     |       Pj#3      |     QKVgen#2    |       Pj#1      |    QKVgen#3     |
// | SA2 |    FFN2s#1      |    FFN1s#2      |      FFN2s#2    |    FFN1s#3      |    FFN2s#3      |    FFN1s#1      |
// | PIM | logit_softmax#3 |    attend#3     | logit_softmax#1 |    attend#1     | logit_softmax#2 |    attend#2     |

// |     |                                  end loop (5 stages)                                    | 
// |     |        L        |        M        |        N        |       O         |        P        |
// |-----|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|
// | SA1 |      Pj#2       |        -        |       Pj#3      |        -        |        -        |
// | SA2 |    FFN2s#1      |    FFN1s#2      |    FFN2s#2      |    FFN1s#3      |    FFN2s#3      |
// | PIM | logit_softmax#3 |    attend#3     |        -        |        -        |        -        |

#ifdef TRI
void StageProgram::init_program() {
    assert(_stage != Stage::Finish);

    if (_breq->_reqs.size() == 0) {
        std::string yellow = "\033[1;33m";
        std::string reset = "\033[0m";
        spdlog::info("{}No request in this batch skip{}", yellow, reset);
        return;
    }

    if (_stage_platform == StagePlatform::PIM) {
        if (skip_PIM_stage()) {
            std::string yellow = "\033[1;33m";
            std::string reset = "\033[0m";
            spdlog::info("{}PIM: skip{}", yellow, reset);
            return;
        } else
            init_PIM_program();
    } else if (_stage_platform == StagePlatform::SA1) {
        if (skip_SA1_stage()) {
            std::string blue = "\033[1;34m";
            std::string reset = "\033[0m";
            spdlog::info("{}SA1: skip{}", blue, reset);
            return;
        } else
            init_SA1_program();
    } else if (_stage_platform == StagePlatform::SA2) {
        if (skip_SA2_stage()) {
            std::string cyan = "\033[1;36m";
            std::string reset = "\033[0m";
            spdlog::info("{}SA2: skip{}", cyan, reset);
            return;
        } else
            init_SA2_program();
    }
}
#else
void StageProgram::init_program() {
    assert(_stage != Stage::Finish);

    if (_breq->_reqs.size() == 0) {
        std::string yellow = "\033[1;33m";
        std::string reset = "\033[0m";
        spdlog::info("{}No request in this batch skip{}", yellow, reset);
        return;
    }

    if (_stage_platform == StagePlatform::PIM) {
        if (skip_pim_stage()) {
            std::string yellow = "\033[1;33m";
            std::string reset = "\033[0m";
            spdlog::info("{}PIM: skip{}", yellow, reset);
            return;
        } else
            init_PIM_program();
    } else if (_stage_platform == StagePlatform::SA)
        init_SA_program();
}
#endif
//////////////////////////////////////
// #1. Stage-related Condition Checks
//////////////////////////////////////

#ifdef TRI
// Refactored Conditional Functions (skip conditions)
bool StageProgram::skip_PIM_stage() {
    return (_stage == Stage::A || _stage == Stage::N || _stage == Stage::O || _stage == Stage::P);
}

bool StageProgram::skip_SA1_stage() {
    return (_stage == Stage::E || _stage == Stage::M || _stage == Stage::O || _stage == Stage::P);
}

bool StageProgram::skip_SA2_stage() {
    return (_stage == Stage::A || _stage == Stage::B || _stage == Stage::C || _stage == Stage::D);
}

// Refactored Conditional Functions (enable conditions)
bool StageProgram::enable_qkv_gen() {
    return (_stage == Stage::A || _stage == Stage::B || _stage == Stage::C) ||
           (_stage == Stage::G || _stage == Stage::I || _stage == Stage::K);
}

bool StageProgram::enable_proj() {
    return (_stage == Stage::D || _stage == Stage::F || _stage == Stage::H) ||
           (_stage == Stage::J || _stage == Stage::L || _stage == Stage::N);
}

bool StageProgram::enable_ffn1s() {
    return (_stage == Stage::E || _stage == Stage::G || _stage == Stage::I) ||
           (_stage == Stage::K || _stage == Stage::M || _stage == Stage::O);
}

bool StageProgram::enable_ffn2s() {
    return (_stage == Stage::F || _stage == Stage::H || _stage == Stage::J) ||
           (_stage == Stage::L || _stage == Stage::N || _stage == Stage::P);
}

bool StageProgram::enable_logit_softmax() {
    return (_stage == Stage::B || _stage == Stage::D || _stage == Stage::F) || 
           (_stage == Stage::H || _stage == Stage::J || _stage == Stage::L); 
}

bool StageProgram::enable_attend() {
    return (_stage == Stage::C || _stage == Stage::E || _stage == Stage::G) || 
           (_stage == Stage::I || _stage == Stage::K || _stage == Stage::M); 
}
#else
bool StageProgram::skip_pim_stage() { return _stage == Stage::A || _stage == Stage::F; }

bool StageProgram::enable_proj_ffns() {
    return _stage == Stage::C || _stage == Stage::D || _stage == Stage::E || _stage == Stage::F;
}

bool StageProgram::enable_qkv_gen() {
    return _stage == Stage::A || _stage == Stage::B || _stage == Stage::C || _stage == Stage::D;
}
#endif


#ifdef TRI
void StageProgram::init_SA1_program() {
    spdlog::info(">>> Initialize SystolicArray (SA1) Stage Model Program <<<");
    auto N = _breq->get_num_rows();
    auto E = Config::global_config.model_n_embd;

    bool lets_proj = enable_proj();
    bool lets_qkvgen = enable_qkv_gen();

    std::vector<uint32_t> input_dim{N, E};
    auto input = std::make_shared<NPUTensor>("SA1_input", input_dim, NPUTensorBufType::ACT, true);
    std::vector<Ptr<BTensor>> inputs{input};

    if (lets_proj) {
        // SA1 handles Projection
        inputs = projection_block(inputs);
        spdlog::info("SA1: Projection enabled");
    }

    if (lets_qkvgen) {
        // SA1 handles QKV Generation
        inputs = qkv_gen_block(inputs);
        spdlog::info("SA1: QKV Generation enabled");
    }

    find_executable_node(input);
}

void StageProgram::init_SA2_program() {
    spdlog::info(">>> Initialize SystolicArray (SA2) Stage Model Program <<<");
    auto N = _breq->get_num_rows();
    auto E = Config::global_config.model_n_embd;

    bool lets_ffn1s = enable_ffn1s();
    bool lets_ffn2s = enable_ffn2s();

    std::vector<uint32_t> input_dim{N, E};
    if (lets_ffn1s || lets_ffn2s) {
        input_dim[1] /= Config::global_config.n_tp;
    }
    auto input = std::make_shared<NPUTensor>("SA2_input", input_dim, NPUTensorBufType::ACT, true);
    std::vector<Ptr<BTensor>> inputs{input};

    if (lets_ffn1s) {
        // SA2 handles FFN1
        inputs = ffn1_block(inputs);
        spdlog::info("SA2: FFN1 enabled");
    }

    if (lets_ffn2s) {
        // SA2 handles FFN2
        inputs = ffn2_block(inputs);
        spdlog::info("SA2: FFN2 enabled");
    }

    find_executable_node(input);
}

void StageProgram::init_PIM_program() {
    spdlog::info(">>> Initialize PIM Stage Model Program <<<");

    uint32_t num_heads = Config::global_config.model_n_head / Config::global_config.n_tp;
    uint32_t dk = Config::global_config.model_n_embd / Config::global_config.model_n_head;
    std::vector<Ptr<BTensor>> querys;
    std::vector<Ptr<BTensor>> keys;
    std::vector<Ptr<BTensor>> values;

    for (auto& request : _breq->_reqs) {
        auto query = std::make_shared<NPUTensor>("query", std::vector<uint32_t>{num_heads, 1, dk},
                                                 NPUTensorBufType::ACT, true);
        querys.push_back(query);
        keys.push_back(request->K_cache[0]);
        values.push_back(request->V_cache[0]);
    }

    // Logit Softmax
    if (enable_logit_softmax()) {
        auto logit_softmax = add_op(std::make_shared<NeuPIMSLogitSoftmax>(
            name_gen(LAYER(0), BlockType::Attention, OperationType::NeuPIMSLogitSoftmax)));
        querys = get_outputs(logit_softmax, querys);
        spdlog::info("PIM: Logit Softmax enabled");
    }

    // Attend
    if (enable_attend()) {
        auto attend = add_op(std::make_shared<NeuPIMSAttend>(
            name_gen(LAYER(0), BlockType::Attention, OperationType::NeuPIMSAttend)));
        querys = get_outputs(attend, querys);
        spdlog::info("PIM: Attend enabled");
    }

    for (auto& query : querys) {
        find_executable_node(query);
    }
}
#else
void StageProgram::init_SA_program() {
    spdlog::info(">>> Initialize SystolicArray Stage Model Program <<<");
    auto N = _breq->get_num_rows();
    auto E = Config::global_config.model_n_embd;

    bool lets_proj_ffns = enable_proj_ffns();
    bool lets_qkvgen = enable_qkv_gen();

    std::vector<uint32_t> input_dim{N, E};
    if (lets_proj_ffns) {
        input_dim[1] /= Config::global_config.n_tp;
    }
    auto input = std::make_shared<NPUTensor>("input", input_dim, NPUTensorBufType::ACT, true);
    std::vector<Ptr<BTensor>> inputs{input};

    if (lets_proj_ffns) {
        // >>> Stage: C/D/E/F : Projection + FFN1 + FFN2
        inputs = projection_block(inputs);
        inputs = ffn1_block(inputs);  // FFN1 & FFN2
        std::string yellow = "\033[1;33m";
        std::string reset = "\033[0m";
        spdlog::info("{}SA : Projection + FFN1 + FFN2{}", yellow, reset);
        // <<< Stage: C/D/E/F
    }

    if (lets_qkvgen) {
        // >>> Stage: A/B/C/D : QKVGen
        inputs = qkv_gen_block(inputs);

        std::string yellow = "\033[1;33m";
        std::string reset = "\033[0m";
        spdlog::info("{}SA : QKV generation{}", yellow, reset);
        // <<< Stage:: A/B/C/D
    }

    find_executable_node(input);
}

void StageProgram::init_PIM_program() {
    spdlog::info(">>> Initialize PIM Stage Model Program <<<");
    std::string yellow = "\033[1;33m";
    std::string reset = "\033[0m";
    spdlog::info("{}PIM: MHA{}", yellow, reset);
    Ptr<NPUTensor> query;
    std::vector<Ptr<BTensor>> inputs;

    int sub_batch_size = _breq->_reqs.size();

    uint32_t num_heads = Config::global_config.model_n_head / Config::global_config.n_tp;
    uint32_t dk = Config::global_config.model_n_embd / Config::global_config.model_n_head;  // 64;

    std::vector<Ptr<BTensor>> querys;
    std::vector<Ptr<BTensor>> keys;
    std::vector<Ptr<BTensor>> values;

    for (int j = 0; j < sub_batch_size; j++) {
        /* - [] todo: change query to real query from gkv gen */
        Ptr<InferRequest> request = _breq->_reqs[j];
        int q_len = request->is_initiated ? 1 : request->input_size;
        assert(q_len == 1);

        query = std::make_shared<NPUTensor>("query", std::vector<uint32_t>{num_heads, q_len, dk},
                                            NPUTensorBufType::ACT, true);
        querys.push_back(query);

        /* key/value cache */
        keys.push_back(request->K_cache[0]);
        values.push_back(request->V_cache[0]);
    }

    /* gemv + softmax */
    std::vector<Ptr<BTensor>> mha_pim_inputs = querys;
    mha_pim_inputs.insert(mha_pim_inputs.end(), keys.begin(),
                          keys.end());  // querys, keys

    auto logit_softmax = add_op(std::make_shared<NeuPIMSLogitSoftmax>(
        name_gen(LAYER(0), BlockType::Attention, OperationType::NeuPIMSLogitSoftmax)));
    inputs = get_outputs(logit_softmax, mha_pim_inputs);

    /* pim_gemv + add */
    inputs.insert(inputs.end(), values.begin(), values.end());  // logits, values

    auto attend = add_op(std::make_shared<NeuPIMSAttend>(
        name_gen(LAYER(0), BlockType::Attention, OperationType::NeuPIMSAttend)));
    inputs = get_outputs(attend, inputs);

    find_executable_node(query);
}
#endif

Ptr<Operation> StageProgram::add_op(std::shared_ptr<Operation> op) {
    // spdlog::info("operation {} added. add_op", op->get_name());
    _op_map[op->get_id()] = op;
    return op;
}

std::vector<Ptr<BTensor>> StageProgram::get_outputs(Ptr<Operation> op,
                                                    std::vector<Ptr<BTensor>> inputs) {
    return op->get_outputs(inputs);
}

void StageProgram::find_executable_node(Ptr<BTensor> tensor) {
    for (auto op : tensor->get_child_nodes()) {
        // spdlog::info("initializing operation {} ...", op->get_name());
        if (op->check_executable()) {
            _executable_operations.push_back(op);
        }
    }
}

bool StageProgram::check_exist_in_executable(uint32_t op_id) {
    for (auto iter = _executable_operations.begin(); iter != _executable_operations.end(); iter++) {
        if (op_id == (*iter)->get_id()) {
            return true;
        }
    }
    return false;
}

void StageProgram::finish_operation(uint32_t id) {
    _op_map[id]->set_finish();
    for (auto iter = _executable_operations.begin(); iter != _executable_operations.end(); iter++) {
        // spdlog::info("iterating operation: {}", (*iter)->get_name());
        if (id == (*iter)->get_id()) {
            // spdlog::info("erasing operation: {}", (*iter)->get_name());
            _executable_operations.erase(iter);
            break;
        }
    }

    for (auto op : _op_map[id]->get_child_nodes()) {
        // spdlog::info("finding operation: {} / {} ", op->get_name(), op->get_id());
        if (op->check_executable() && !check_exist_in_executable(op->get_id())) {
            // spdlog::info("found operation: {}", op->get_name());
            _executable_operations.push_back(op);
        }
    }
}

bool StageProgram::check_finish() {
    bool finish = true;
    for (auto const &[key, val] : _op_map) {
        finish = finish && val->check_finish();
    }

    return finish;
}

std::vector<OperationStat> StageProgram::list_operation_stat() {
    std::vector<OperationStat> ret;
    for (auto &[key, val] : _op_map) {
        ret.push_back(val->get_stat());
    }

    return ret;
}

void StageProgram::finish_operation_tile(Tile &tile) {
    _op_map[tile.operation_id]->reduce_tile(tile);
}

/**
 * logger function forStageProgram
 * TODO: log file name is tentative. think of fname rule
 */
void StageProgram::log() {
    std::string fname = Config::global_config.log_dir + "/" + _name;
    Logger::log(list_operation_stat(), fname);
}
#ifdef TRI
std::vector<Ptr<BTensor>> StageProgram::projection_block(std::vector<Ptr<BTensor>> inputs) {
    auto N = _breq->get_num_rows();
    auto E = Config::global_config.model_n_embd;

    std::vector<uint32_t> input_dim{N, E};
    auto res_buf =
        std::make_shared<NPUTensor>("residual_buffer", input_dim, NPUTensorBufType::ACT, true);

    int layer = 0;
    auto prefix = name_gen(LAYER(0), BlockType::Attention);
    // auto res_buf = inputs[0];

    auto projection = add_op(std::make_shared<MatMul>(
        name_gen(prefix, OperationType::Projection),
        _model->get_params(layer, BlockType::Attention, OperationType::Projection)));
    inputs = get_outputs(projection, inputs);

    // fixme: residual is not with this tensor.
    auto residual = add_op(std::make_shared<Add>(name_gen(prefix, OperationType::Residual)));
    inputs.push_back(res_buf);
    inputs = get_outputs(residual, inputs);
    return inputs;
}

// ffn1_block(original): LayerNorm -> MatMul(fc1) -> Gelu -> MatMul(fc2) -> Add
// ffn1_block(new): FFN1 (LayerNorm -> MatMul(fc1) -> Gelu)
// ffn2_block(new): FFN2 (FFN1-> MatMul(fc2) -> Add)
std::vector<Ptr<BTensor>> StageProgram::ffn1_block(std::vector<Ptr<BTensor>> inputs) {
    int layer = 0;
    std::string prefix = name_gen(LAYER(layer), BlockType::FeedForward);
    
    // LayerNorm
    auto ln = add_op(std::make_shared<LayerNorm>(
        name_gen(prefix, OperationType::LayerNorm),
        _model->get_params(layer, BlockType::FeedForward, OperationType::LayerNorm)));
    inputs = get_outputs(ln, inputs);

    // Fully Connected 1
    auto fc1 = add_op(std::make_shared<MatMul>(
        name_gen(prefix, OperationType::FullyConnected1),
        _model->get_params(layer, BlockType::FeedForward, OperationType::FullyConnected1)));
    inputs = get_outputs(fc1, inputs);

    // Gelu activation
    auto gelu = add_op(std::make_shared<Gelu>(name_gen(prefix, OperationType::Gelu)));
    inputs = get_outputs(gelu, inputs);

    return inputs;
}

// ffn2_block(new): FFN2 (FFN1 -> MatMul(fc2) -> Add)
std::vector<Ptr<BTensor>> StageProgram::ffn2_block(std::vector<Ptr<BTensor>> inputs) {
    int layer = 0;
    auto res_buf = inputs[0];  // original residual buffer
    
    std::string prefix = name_gen(LAYER(layer), BlockType::FeedForward);

    // Fully Connected 2
    auto fc2 = add_op(std::make_shared<MatMul>(
        name_gen(prefix, OperationType::FullyConnected2),
        _model->get_params(layer, BlockType::FeedForward, OperationType::FullyConnected2)));
    inputs = get_outputs(fc2, inputs);

    // Residual connection (Add)
    auto residual = add_op(std::make_shared<Add>(name_gen(prefix, OperationType::Residual)));
    inputs.push_back(res_buf);
    inputs = get_outputs(residual, inputs);

    return inputs;
}

std::vector<Ptr<BTensor>> StageProgram::qkv_gen_block(std::vector<Ptr<BTensor>> inputs) {
    int layer = 0;
    auto prefix = name_gen(LAYER(0), BlockType::Attention);

    // (N,E) -> (N,E)
    auto ln1 = add_op(std::make_shared<LayerNorm>(
        name_gen(prefix, OperationType::LayerNorm),
        _model->get_params(layer, BlockType::Attention, OperationType::LayerNorm)));
    inputs = get_outputs(ln1, inputs);

    // (N,E) x (E,3E)
    auto qkv_gen = add_op(std::make_shared<MatMul>(
        name_gen(prefix, OperationType::QKVGen),
        _model->get_params(layer, BlockType::Attention, OperationType::QKVGen)));
    inputs = get_outputs(qkv_gen, inputs);

    return inputs;
}
#else
std::vector<Ptr<BTensor>> StageProgram::projection_block(std::vector<Ptr<BTensor>> inputs) {
    auto N = _breq->get_num_rows();
    auto E = Config::global_config.model_n_embd;

    std::vector<uint32_t> input_dim{N, E};
    auto res_buf =
        std::make_shared<NPUTensor>("residual_buffer", input_dim, NPUTensorBufType::ACT, true);

    int layer = 0;
    auto prefix = name_gen(LAYER(0), BlockType::Attention);
    // auto res_buf = inputs[0];

    auto projection = add_op(std::make_shared<MatMul>(
        name_gen(prefix, OperationType::Projection),
        _model->get_params(layer, BlockType::Attention, OperationType::Projection)));
    inputs = get_outputs(projection, inputs);

    // fixme: residual is not with this tensor.
    auto residual = add_op(std::make_shared<Add>(name_gen(prefix, OperationType::Residual)));
    inputs.push_back(res_buf);
    inputs = get_outputs(residual, inputs);
    return inputs;
}
std::vector<Ptr<BTensor>> StageProgram::ffn1_block(std::vector<Ptr<BTensor>> inputs) {
    int layer = 0;
    auto res_buf = inputs[0];
    std::string prefix = name_gen(LAYER(layer), BlockType::FeedForward);
    // create operations
    auto ln = add_op(std::make_shared<LayerNorm>(
        name_gen(prefix, OperationType::LayerNorm),
        _model->get_params(layer, BlockType::FeedForward, OperationType::LayerNorm)));
    inputs = get_outputs(ln, inputs);

    auto fc1 = add_op(std::make_shared<MatMul>(
        name_gen(prefix, OperationType::FullyConnected1),
        _model->get_params(layer, BlockType::FeedForward, OperationType::FullyConnected1)));
    inputs = get_outputs(fc1, inputs);

    auto gelu = add_op(std::make_shared<Gelu>(name_gen(prefix, OperationType::Gelu)));
    inputs = get_outputs(gelu, inputs);

    auto fc2 = add_op(std::make_shared<MatMul>(
        name_gen(prefix, OperationType::FullyConnected2),
        _model->get_params(layer, BlockType::FeedForward, OperationType::FullyConnected2)));
    inputs = get_outputs(fc2, inputs);

    auto residual = add_op(std::make_shared<Add>(name_gen(prefix, OperationType::Residual)));
    inputs.push_back(res_buf);
    inputs = get_outputs(residual, inputs);
    return inputs;
}
std::vector<Ptr<BTensor>> StageProgram::ffn2_block(std::vector<Ptr<BTensor>> inputs) {
    // ffn1_block includes ffn2
    return inputs;
}

std::vector<Ptr<BTensor>> StageProgram::qkv_gen_block(std::vector<Ptr<BTensor>> inputs) {
    int layer = 0;
    auto prefix = name_gen(LAYER(0), BlockType::Attention);

    // (N,E) -> (N,E)
    auto ln1 = add_op(std::make_shared<LayerNorm>(
        name_gen(prefix, OperationType::LayerNorm),
        _model->get_params(layer, BlockType::Attention, OperationType::LayerNorm)));
    inputs = get_outputs(ln1, inputs);

    // (N,E) x (E,3E)
    auto qkv_gen = add_op(std::make_shared<MatMul>(
        name_gen(prefix, OperationType::QKVGen),
        _model->get_params(layer, BlockType::Attention, OperationType::QKVGen)));
    inputs = get_outputs(qkv_gen, inputs);

    return inputs;
}
#endif