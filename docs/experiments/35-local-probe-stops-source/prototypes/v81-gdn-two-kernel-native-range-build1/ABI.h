#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Arrays: activated,g,beta,S,q_ready,k_ready,v_ready,scales,o,S_new. */
int v81_register_bank(uint32_t arm,uint32_t bank,const uint64_t* addresses,const uint64_t* sizes);
int v81_build_ranges(void);
int v81_loop_range(uint32_t arm,uint32_t groups);
int v81_copy_output(uint32_t arm,uint32_t bank,uint32_t output,uint64_t size,void* destination);
int v81_verify_readonly(void);
const char* v81_report_json(void);
#ifdef __cplusplus
}
#endif
