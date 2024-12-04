#pragma once

#include <string>
#include <vector>

#include "Common.h"

struct Mapping {
    enum LoopName { N, C, M, S, R, Q, P };
    struct LoopCounts {
        uint32_t N = 1; // Batch size
        uint32_t C = 1; // Number of input channles
        uint32_t M = 1; // Number of output channles
        uint32_t S = 1; // Weight height
        uint32_t R = 1; // Weight width
        uint32_t Q = 1; // INput height
        uint32_t P = 1; // Input width
        uint32_t target_core = 0;
        bool operator==(const LoopCounts &other) const {
            return (N == other.N) && (C == other.C) && (M == other.M) && (S == other.S) &&
                   (R == other.R) && (Q == other.Q) && (P == other.P);
        }
        bool operator<(const LoopCounts &other) const {
            if (N < other.N)
                return true;
            else if (N > other.N)
                return false;
            if (C < other.C)
                return true;
            else if (C > other.C)
                return false;
            if (M < other.M)
                return true;
            else if (M > other.M)
                return false;
            if (S < other.S)
                return true;
            else if (S > other.S)
                return false;
            if (R < other.R)
                return true;
            else if (R > other.R)
                return false;
            if (Q < other.Q)
                return true;
            else if (Q > other.Q)
                return false;
            if (P < other.P)
                return true;
            return false;
        }
        uint32_t get_loop(LoopName name);
    };
    Mapping() {}
    Mapping(std::string mapping_line);
    LoopCounts total_loop;
    LoopCounts tile_in_loop;
    LoopCounts tile_out_loop;
    uint32_t spatial_M = 0;
    uint32_t spatial_P = 0;
    uint32_t spatial_Q = 0;
    uint32_t spatial_C = 0;
    uint32_t spatial_R = 0;
    uint32_t spatial_S = 0;
    std::vector<LoopName> tile_out_loop_order;
};
typedef std::map<Mapping::LoopCounts, Mapping> MappingTable;
MappingTable parse_mapping_file(std::string file_path);
MappingTable from_config(SimulationConfig config);
