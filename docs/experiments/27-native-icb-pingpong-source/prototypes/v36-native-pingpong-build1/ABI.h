#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int v33_pin(uint64_t addr, uint64_t nbytes, const char *parameter_name);
int v33_begin(void);
int v33_end_build(void);
int v33_register_output(uint64_t original_addr, uint64_t nbytes, const char *label);
int v33_restore_replay(void);
int v33_copy_output(uint64_t original_addr, uint64_t nbytes, void *dst);
const char *v33_report_json(void);
#ifdef __cplusplus
}
#endif
#ifdef __cplusplus
extern "C" {
#endif
// Input role: 0=state, 1=token, 2=position, 3=lens. All scalar roles are int32[1].
int v34_register_input(uint64_t addr,uint64_t nbytes,uint32_t role,const char *label);
int v34_link(uint64_t input_addr,uint64_t output_addr,uint64_t nbytes);
int v34_seal_feedback(uint32_t state_count,uint32_t capacity);
int v34_reset(void);
int v34_write_input(uint64_t original_input_addr,uint64_t nbytes,const void *source);
int v34_replay_step(void);
int v34_compare_output(uint64_t original_output_addr,uint64_t nbytes,const void *reference);
int v34_copy_output(uint64_t original_output_addr,uint64_t nbytes,void *destination);
const char *v34_report_json(void);
#ifdef __cplusplus
}
#endif
#ifdef __cplusplus
extern "C" {
#endif
// Ineligible requested pairs return 0, retain staged fallback, and report reason.
// A caller claiming S-pingpong must assert actual eligible_count after registration.
int v36_register_pingpong(uint64_t input_addr,uint64_t output_addr,uint64_t nbytes);
// Export only after a request. No JSON serialization occurs in replay_step.
const char *v36_timing_json(void);
#ifdef __cplusplus
}
#endif
