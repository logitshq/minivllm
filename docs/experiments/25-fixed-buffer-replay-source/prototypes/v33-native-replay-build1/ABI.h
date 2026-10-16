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
