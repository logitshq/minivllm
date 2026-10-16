#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int v80_register_slot(uint32_t arm,uint32_t slot,uint64_t w,uint64_t wn,uint64_t scales,uint64_t sn,uint64_t x,uint64_t xn,uint64_t y,uint64_t yn);
int v80_build_ranges(void);
int v80_loop_range(uint32_t arm,uint32_t groups);
int v80_copy_output(uint32_t arm,uint32_t slot,uint64_t n,void* destination);
int v80_verify_readonly(void);
const char* v80_report_json(void);
#ifdef __cplusplus
}
#endif
