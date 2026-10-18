python3 -I -B - <<'PY'
import importlib.abc,sys,unittest
sys.path.insert(0,'/Users/ryanli/code/minivllm')
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'mlx','torch','numpy','tutorial_code','mlx_lm','mlx_vlm','omlx'}: raise AssertionError('forbidden import '+name)
sys.meta_path.insert(0,Block())
def guard(event,args):
 if event in {'subprocess.Popen','os.system','os.kill','os.killpg','socket.connect','socket.bind','socket.getaddrinfo'}: raise AssertionError('forbidden effect '+event)
sys.addaudithook(guard)
suite=unittest.defaultTestLoader.loadTestsFromName('bench.m5_ultra_256gb.c1_pipeline_profile.test_instrumentation_cpu')
print('discovered',suite.countTestCases())
result=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(not result.wasSuccessful())
PY
