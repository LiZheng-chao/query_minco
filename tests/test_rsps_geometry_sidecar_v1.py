"""Portable semantic negatives plus explicitly requested real-sidecar qualification."""
import copy,json,os,pathlib,tempfile,unittest
import torch
from query_minco.training.datasets.geometry_sidecar import GeometrySidecar,validate_semantics
from query_minco.training.datasets.manifest import FrozenDataset
from query_minco.training.io import sha

class Semantics(unittest.TestCase):
    def setUp(self):
        self.row=dict(split='train',frame='fixed B0 FLU',units='m_s',domain_semantics='center_bounds_not_occupied_solids',empty_certified=False,body_radius_m=.12,safety_margin_m=.03,flight_center_domain_m=[[-1,-1,-1],[1,1,1]],obstacles=[dict(name='box',min=[2.,-1.,-1.],max=[3.,1.,1.])])
    def test_valid_and_empty(self):
        validate_semantics(self.row)
        self.row['obstacles']=[]
        with self.assertRaises(ValueError):validate_semantics(self.row)
        self.row['empty_certified']=True;validate_semantics(self.row)
    def test_illegal_geometry(self):
        for key,value in [('split','eval'),('frame','world'),('units','mm'),('body_radius_m',.1),('domain_semantics','occupied_solids'),('flight_center_domain_m',[[1,1,1],[-1,-1,-1]])]:
            row=copy.deepcopy(self.row);row[key]=value
            with self.assertRaises(ValueError):validate_semantics(row)
        for lo,hi in [([0,0,0],[0,0,0]),([float('nan'),0,0],[1,1,1]),(['0',0,0],[1,1,1])]:
            row=copy.deepcopy(self.row);row['obstacles'][0].update(min=lo,max=hi)
            with self.assertRaises(ValueError):validate_semantics(row)

@unittest.skipUnless(os.environ.get('RSPS_SIDECAR'),'real data only via explicit environment binding')
class QualifiedReal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path=pathlib.Path(os.environ['RSPS_SIDECAR']);cls.m=json.loads(cls.path.read_text())
        cls.ds=FrozenDataset(cls.m['dataset'],'b2c1_reference_v1')
        cls.loader=GeometrySidecar(dict(path=str(cls.path),sha256=sha(cls.path)),cls.ds)
    def reject(self,mutate):
        m=copy.deepcopy(self.m);mutate(m)
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d)/'negative.json';p.write_text(json.dumps(m))
            with self.assertRaises((ValueError,KeyError)):GeometrySidecar(dict(path=str(p),sha256=sha(p)),self.ds)
    def test_all_384_and_three_seeds(self):
        for sid in self.ds.rows:
            samples=self.ds.load([sid]);bound=self.loader.load(samples)
            self.assertEqual(len(bound),1);self.assertFalse(bound[0].boxes.requires_grad)
        self.assertEqual(set(self.loader.orders),{42,43,44})
        print('QUALIFIED_REAL_SIDECAR=384/384; SEEDS=42,43,44')
    def test_reject_split_hash_shape_source(self):
        self.reject(lambda m:m['rows'][0].update(split='eval'))
        self.reject(lambda m:m['rows'][0]['scene_source'].update(sha256='0'*64))
        self.reject(lambda m:m['rows'][0].update(scene_pointer='/999999/scene'))
        self.reject(lambda m:m['rows'][0]['obstacles'][0].update(min=[1,2]))
        self.reject(lambda m:m['rows'].append(copy.deepcopy(m['rows'][0])))
        self.reject(lambda m:m['rows'].pop())
        self.reject(lambda m:m['rows'][0].update(sample_id='eval_goal_f00_fake'))
        self.reject(lambda m:m['orders'].pop())
    def test_reject_forged_sample_and_binding(self):
        samples=self.ds.load([next(iter(self.ds.rows))]);samples[0].state[0]+=.1
        with self.assertRaises(ValueError):self.loader.load(samples)
        with self.assertRaises(ValueError):GeometrySidecar(dict(path=str(self.path),sha256='0'*64),self.ds)
if __name__=='__main__':unittest.main(verbosity=2)
