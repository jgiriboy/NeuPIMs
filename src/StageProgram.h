#pragma once

#include <vector>
#include <stdexcept> // if not included before, needed for runtime_error
#include "BatchedRequest.h"
#include "Common.h"
#include "Logger.h"
#include "Model.h"
#include "Stat.h"
#include "operations/Operation.h"
#include "tensor/BTensor.h"

class StageProgram {
   public:
    StageProgram(std::shared_ptr<Model> model, Ptr<BatchedRequest> batched_request,
                 StagePlatform stage_type, Stage stage);
    void init_program();
    Ptr<Operation> add_op(Ptr<Operation> op);
    std::vector<Ptr<BTensor>> get_outputs(Ptr<Operation> op, std::vector<Ptr<BTensor>> inputs);

    bool check_exist_in_executable(uint32_t op_id);
    void finish_operation(uint32_t id);
    void find_executable_node(Ptr<BTensor> tensor);
    std::vector<std::shared_ptr<Operation>> get_executable_operations() {
        return _executable_operations;
    }
    bool check_finish();
    std::vector<OperationStat> list_operation_stat();
    void finish_operation_tile(Tile& tile);
    void log();

    std::string _name;

    std::shared_ptr<Model> _model;
    std::shared_ptr<BatchedRequest> _breq;
    robin_hood::unordered_map<uint32_t, Ptr<Operation>> _op_map;
    std::map<uint32_t, Ptr<BTensor>> _tensor_map;
    std::vector<std::shared_ptr<Operation>> _executable_operations;

    StagePlatform _stage_platform;
    Stage _stage;

    void init_SA1_program();
    void init_SA2_program();
    void init_PIM_program();

 // Conditional Checks
    bool skip_PIM_stage();
    bool skip_SA1_stage();
    bool skip_SA2_stage();
    bool enable_qkv_gen();
    bool enable_proj();
    bool enable_ffn1s();
    bool enable_ffn2s();
    bool enable_logit_softmax();
    bool enable_attend();

    std::vector<Ptr<BTensor>> projection_block(std::vector<Ptr<BTensor>> inputs);
    std::vector<Ptr<BTensor>> ffn1_block(std::vector<Ptr<BTensor>> inputs);
    std::vector<Ptr<BTensor>> ffn2_block(std::vector<Ptr<BTensor>> inputs);
    std::vector<Ptr<BTensor>> qkv_gen_block(std::vector<Ptr<BTensor>> inputs);
};