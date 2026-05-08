#pragma once

#include "llama.h"

#include <unordered_map>
#include <vector>

// Cross-ubatch shift state: pair (h_p, x_{p+1}) at MTP pos p+1. The last
// h-row of one ubatch needs the first token of the NEXT ubatch to pair
// with, so it's stashed here until that next ubatch fires. Resets when
// pos_start of the new ubatch != pending_pos+1 (new prompt or seq_rm gap).
// One entry per active sequence (slot) so concurrent --parallel slots don't
// stomp on each other.
struct llama_mtp_seq_state {
    std::vector<float> pending_h;
    llama_pos          pending_pos = -1;
};

struct llama_mtp {
    llama_context * ctx_mtp    = nullptr; // non-owning
    llama_batch     hook_batch = {};      // sized to n_ubatch

    std::unordered_map<llama_seq_id, llama_mtp_seq_state> seq_states;
};
