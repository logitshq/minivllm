import math, unittest
from policy import choose_width, survival_from_full_width

class TestPolicy(unittest.TestCase):
    def test_short_wins(self):
        # survival decays fast after 3, long cost not justified
        survival=[0.5,0.1,0.05,0.01,0.01,0.005,0.0]
        costs={4:100,8:200}
        self.assertEqual(choose_width(survival,costs,100),4)
    def test_long_wins(self):
        survival=[0.9,0.8,0.7,0.6,0.5,0.4,0.3]
        costs={4:100,8:120}
        self.assertEqual(choose_width(survival,costs,100),8)
    def test_output_cap(self):
        survival=[0.9]*7
        costs={4:100,8:150}
        # remaining=2 => both widths cap to take 1 -> same expected 1.9, cheaper wins (4)
        self.assertEqual(choose_width(survival,costs,2),4)
        # remaining=1 => expected 1 both, tie smaller
        self.assertEqual(choose_width(survival,costs,1),4)
    def test_equality_tie(self):
        survival=[0.0]*7
        costs={4:100,8:100}
        self.assertEqual(choose_width(survival,costs,100),4)
    def test_cumulative_vs_conditional(self):
        # unconditional survival must be used; conditional would overestimate
        # e.g., data [2,2,0] -> survival = [2/3,2/3,0...] expected width4=1+1.333=2.333
        # conditional product misunderstanding would differ
        surv=survival_from_full_width([2,2,0])
        self.assertAlmostEqual(surv[0],2/3)
        self.assertAlmostEqual(surv[1],2/3)
        self.assertAlmostEqual(surv[2],0)
        # choose with tie logic
        costs={4:100,8:200}
        # expected deterministic
        self.assertEqual(choose_width(surv,costs,100),4)

    def test_survival_from_values(self):
        self.assertEqual(survival_from_full_width([0,7,7]),[2/3,2/3,2/3,2/3,2/3,2/3,2/3])

    def test_empty(self):
        with self.assertRaises(ValueError): survival_from_full_width([])
        with self.assertRaises((ValueError,TypeError)): choose_width([],{4:1,8:1},1)

    def test_nan_bool_censored(self):
        with self.assertRaises(TypeError): survival_from_full_width([True])
        with self.assertRaises(TypeError): choose_width([True]*7,{4:10,8:10},5)
        with self.assertRaises(ValueError): choose_width([float('nan')]+[0]*6,{4:10,8:10},5)
        with self.assertRaises(ValueError): choose_width([0.5]*7,{4:0,8:10},5)
        with self.assertRaises(TypeError): survival_from_full_width([1.5])
        with self.assertRaises(TypeError): choose_width([0.5]*7,{4:10,8:10},True)
        with self.assertRaises(ValueError): choose_width([1.0,0.9,1.0,0,0,0,0],{4:10,8:10},5) # increasing
        with self.assertRaises(ValueError): survival_from_full_width([8])
        with self.assertRaises(ValueError): choose_width([0.5]*7,{4:10,8:10},0)
        with self.assertRaises(ValueError): choose_width([0.5]*7,{4:10},5)
        with self.assertRaises(ValueError): choose_width([0.5]*7,{4:10,8:10,2:5},5)
        with self.assertRaises(TypeError): choose_width([0.5]*7,{4:True,8:10},5)

if __name__=="__main__":
    unittest.main()
