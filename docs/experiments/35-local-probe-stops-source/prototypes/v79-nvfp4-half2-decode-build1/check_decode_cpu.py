"""Exhaustive integer/IEEE-half encoding proof; no MLX/GPU."""
import struct,ast,json
from pathlib import Path
D=Path(__file__).resolve().parent
MAG=(0.,.5,1.,1.5,2.,3.,4.,6.)
def paired(c):
 u=(c&15)|((c&240)<<12);m=u&0x00070007
 nz=(m|(m>>1)|(m>>2))&0x00010001;ge=((m>>1)|(m>>2))&0x00010001
 return (nz*0x3600+m*0x200+ge*0x200+((u&0x00080008)<<12))&0xffffffff
expected=[struct.unpack('<H',struct.pack('<e',-MAG[c&7] if c&8 else MAG[c&7]))[0] for c in range(16)]
for word in range(65536):
 bits=paired(word&255)|(paired(word>>8)<<32)
 assert [(bits>>(16*i))&65535 for i in range(4)]==[expected[(word>>(4*i))&15] for i in range(4)]
for name in ['geometry.py','micro.py']:ast.parse((D/name).read_text())
print(json.dumps({'status':'PASS_CPU_EXHAUSTIVE_HALF_ENCODINGS','packed_words':65536,'half_words':262144,'signedzero_bits':expected[0:1]+expected[8:9],'GPU_used':False}))
