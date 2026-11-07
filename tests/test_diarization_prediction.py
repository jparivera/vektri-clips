import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('adapter', Path(__file__).resolve().parents[1] / 'scripts' / 'prepare_diarization_prediction.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def evidence():
    source = {'sha256': 'a'*64, 'duration_seconds': 2., 'sample_rate':16000,
              'channels':1,'sample_width_bytes':2,'frames':32000}
    cfg = {'provider':'cpu','num_threads':2,'num_clusters':-1,'cluster_threshold':.5,
           'speaker_count_mode':'unknown','threshold_used':True,'window_shift_ratio':.1,
           'min_duration_on':.3,'min_duration_off':.5}
    models = [{'id':model_id,'filename':'model.onnx','bytes':100,'sha256':'d'*64,
               'revision':'e'*40,'repository':'official/model','license':license_id,
               'path':f'/models/{model_id}/model.onnx'}
              for model_id,license_id in zip(adapter.MODEL_IDS,adapter.MODEL_LICENSES)]
    result = {'source_sha256':'a'*64, 'config':cfg, 'models':models,
              'runtime':{'sherpa_onnx':'2.0.1'}, 'speaker_count':1,
              'turns':[{'start':.1,'end':1.5,'speaker_index':0,'speaker':'speaker_00'}]}
    report = {'schema_version':1,'status':'succeeded','source_unchanged':True,'models_unchanged':True,
              'source':source,'config':cfg,'models':models,'result':result,
              'resources':{'exit_code':0},'runner_sha256':'1'*64,'manifest_sha256':'2'*64,
              'timestamp_boundary_tolerance_seconds':.05}
    gold = {'schema_version':1,'duration_seconds':2.,'audio':{'sha256':'a'*64}}
    return report,gold


class ConversionTests(unittest.TestCase):
    def test_exact_conversion_keeps_original_evidence(self):
        report,gold=evidence(); before=copy.deepcopy((report,gold))
        value=adapter.convert(report,gold,'b'*64,'c'*64)
        self.assertEqual(value['segments'],[{'start':.1,'end':1.5,'speaker':'speaker_00'}])
        self.assertEqual(value['source_sha256'],'a'*64)
        self.assertEqual(value['runner_sha256'],'1'*64)
        self.assertEqual(value['manifest_sha256'],'2'*64)
        self.assertEqual(value['models'],report['models'])
        self.assertEqual(value['conversion_policy']['adjustments'],[])
        self.assertEqual((report,gold),before)

    def test_frozen_boundary_intersection_is_recorded(self):
        report,gold=evidence()
        report['result']['turns'][0]['end']=2.04
        value=adapter.convert(report,gold,'b'*64,'c'*64)
        self.assertEqual(value['segments'][0]['end'],2.)
        self.assertEqual(value['conversion_policy']['adjustments'][0]['raw_end'],2.04)
        report['result']['turns'][0]['start']=2.01
        value=adapter.convert(report,gold,'b'*64,'c'*64)
        self.assertEqual(value['segments'],[])
        self.assertEqual(value['conversion_policy']['adjustments'][0]['action'],'excluded_outside_audio')

    def test_exact_fifty_millisecond_boundary_and_next_float(self):
        report,gold=evidence();turn=report['result']['turns'][0]
        turn['end']=gold['duration_seconds']+.05
        value=adapter.convert(report,gold,'b'*64,'c'*64)
        self.assertEqual(value['segments'][0]['end'],2.)
        self.assertEqual(value['conversion_policy']['adjustments'][0]['raw_end'],2.05)
        turn['end']=math.nextafter(2.05,math.inf)
        with self.assertRaises(ValueError):adapter.convert(report,gold,'b'*64,'c'*64)

    def test_overlap_sparse_ids_preserved_without_merging(self):
        report,gold=evidence()
        report['result']['turns']=[{'start':.1,'end':1.5,'speaker_index':3,'speaker':'speaker_03'},
                                   {'start':.5,'end':1.9,'speaker_index':17,'speaker':'speaker_17'}]
        report['result']['speaker_count']=2
        value=adapter.convert(report,gold,'b'*64,'c'*64)
        self.assertEqual(value['scored_speaker_count'],2)
        self.assertEqual(value['segments'],[{'start':.1,'end':1.5,'speaker':'speaker_03'},
                                            {'start':.5,'end':1.9,'speaker':'speaker_17'}])

    def test_more_than_six_labels_are_preserved_for_scorer_rejection(self):
        report,gold=evidence()
        report['result']['turns']=[{'start':index/10,'end':1.,'speaker_index':index*2,
                                   'speaker':f'speaker_{index*2:02d}'} for index in range(7)]
        report['result']['speaker_count']=7
        value=adapter.convert(report,gold,'b'*64,'c'*64)
        self.assertEqual(value['scored_speaker_count'],7)
        self.assertEqual(len(value['segments']),7)
        scorer_path=Path(__file__).resolve().parents[1]/'scripts'/'score_diarization.py'
        score_spec=importlib.util.spec_from_file_location('score_diarization',scorer_path)
        scorer=importlib.util.module_from_spec(score_spec);score_spec.loader.exec_module(scorer)
        with self.assertRaisesRegex(ValueError,'more than 6 speakers'):
            scorer.validate_segments(value['segments'],2.,'Hypothesis')

    def test_failure_and_binding_mismatches_rejected(self):
        for key,value in [('status','failed'),('source_unchanged',False),('models_unchanged',False),
                          ('timestamp_boundary_tolerance_seconds',.1)]:
            with self.subTest(key=key):
                report,gold=evidence(); report[key]=value
                with self.assertRaises(ValueError):adapter.convert(report,gold,'b'*64,'c'*64)
        report,gold=evidence();gold['audio']['sha256']='d'*64
        with self.assertRaises(ValueError):adapter.convert(report,gold,'b'*64,'c'*64)

    def test_invalid_native_evidence_rejected(self):
        for changes in ({'end':2.06},{'start':-.01},{'end':float('nan')},{'speaker_index':True},
                        {'speaker':'person_name'},{'start':1.5,'end':1.}):
            with self.subTest(changes=changes):
                report,gold=evidence();report['result']['turns'][0].update(changes)
                with self.assertRaises(ValueError):adapter.convert(report,gold,'b'*64,'c'*64)

    def test_missing_or_invalid_provenance_hashes_rejected(self):
        for key in ('runner_sha256','manifest_sha256'):
            for value in (None,'','ABCDEF'*10,'0'*63,'g'*64,True,123):
                with self.subTest(key=key,value=value):
                    report,gold=evidence();report[key]=value
                    with self.assertRaises(ValueError):adapter.convert(report,gold,'b'*64,'c'*64)
            report,gold=evidence();del report[key]
            with self.assertRaises(ValueError):adapter.convert(report,gold,'b'*64,'c'*64)

    def test_failed_or_mistyped_native_exit_is_never_scoreable(self):
        for value in (None,-9,1,'0',False,0.,[],{}):
            with self.subTest(value=value):
                report,gold=evidence();report['resources']['exit_code']=value
                with self.assertRaises(ValueError):adapter.convert(report,gold,'b'*64,'c'*64)
        report,gold=evidence();del report['resources']
        with self.assertRaises(ValueError):adapter.convert(report,gold,'b'*64,'c'*64)

    def test_identical_but_invalid_parent_and_child_config_rejected(self):
        cases=[('provider','cuda'),('num_threads',12),('num_threads',2.),('num_threads',True),
               ('num_clusters',0),('num_clusters',9),('num_clusters',-1.),('num_clusters',True),
               ('cluster_threshold',0),('cluster_threshold',1),('cluster_threshold',float('nan')),
               ('cluster_threshold',True),('speaker_count_mode','explicit_known_count'),
               ('threshold_used',False),('threshold_used',1),('window_shift_ratio',.2),
               ('min_duration_on',.4),('min_duration_off',.6),('unexpected',True)]
        for key,value in cases:
            with self.subTest(key=key,value=value):
                report,gold=evidence();report['config'][key]=value
                self.assertEqual(report['result']['config'],report['config'])
                with self.assertRaises(ValueError):adapter.convert(report,gold,'b'*64,'c'*64)
        report,gold=evidence();del report['config']['num_threads']
        with self.assertRaises(ValueError):adapter.convert(report,gold,'b'*64,'c'*64)

    def test_known_count_requires_consistent_mode_and_unused_threshold(self):
        report,gold=evidence();cfg=report['config']
        cfg.update(num_clusters=3,speaker_count_mode='explicit_known_count',threshold_used=False)
        self.assertEqual(adapter.convert(report,gold,'b'*64,'c'*64)['config'],cfg)
        cfg['threshold_used']=True
        with self.assertRaises(ValueError):adapter.convert(report,gold,'b'*64,'c'*64)

    def test_identical_but_invalid_parent_and_child_model_pins_rejected(self):
        cases=[('id','unknown'),('bytes',0),('bytes',True),('bytes',1.5),('bytes',64*1024**2+1),
               ('sha256',''),('sha256','f'*63),('sha256','G'*64),('revision',''),('revision','main'),
               ('repository',''),('repository','/model'),('repository','org/model/extra'),
               ('filename','../model.onnx'),('filename','model.pt'),('filename',''),
               ('path',''),('path','relative/model.onnx'),('path','/models/other.onnx'),
               ('path','/models/../model.onnx'),('license',''),('license','noncommercial')]
        for key,value in cases:
            with self.subTest(key=key,value=value):
                report,gold=evidence();report['models'][0][key]=value
                self.assertEqual(report['result']['models'],report['models'])
                with self.assertRaises(ValueError):adapter.convert(report,gold,'b'*64,'c'*64)
        for value in ([{},{}],[],[None,None]):
            report,gold=evidence();report['models']=value;report['result']['models']=value
            with self.assertRaises(ValueError):adapter.convert(report,gold,'b'*64,'c'*64)

    def test_valid_but_different_child_provenance_rejected(self):
        for key in ('models','config'):
            report,gold=evidence();report['result'][key]=copy.deepcopy(report[key])
            if key=='models':report['result'][key][0]['sha256']='9'*64
            else:report['result'][key]['cluster_threshold']=.6
            with self.assertRaises(ValueError):adapter.convert(report,gold,'b'*64,'c'*64)

    def test_speaker_count_not_guessed_or_collapsed(self):
        report,gold=evidence();report['result']['speaker_count']=3
        with self.assertRaises(ValueError):adapter.convert(report,gold,'b'*64,'c'*64)
        report,gold=evidence();report['result']['turns']=[];report['result']['speaker_count']=0
        self.assertEqual(adapter.convert(report,gold,'b'*64,'c'*64)['segments'],[])

    def test_bounded_json_refuses_symlink_and_oversize(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'a.json';path.write_text('{}')
            link=Path(tmp)/'link';link.symlink_to(path)
            with self.assertRaises(OSError):adapter.bounded_json(link)
            path.write_bytes(b' '*(adapter.LIMIT+1))
            with self.assertRaises(ValueError):adapter.bounded_json(path)

    def test_nonregular_json_inputs_are_rejected_without_waiting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fifo=root/'pipe';os.mkfifo(fifo)
            for path in (root,fifo):
                with self.subTest(path=path):
                    with self.assertRaises(ValueError):adapter.bounded_json(path)

    def test_nonfinite_json_literals_and_exponents_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.json'
            for value in ('NaN','Infinity','-Infinity','1e309','-1e309'):
                with self.subTest(value=value):
                    path.write_text('{"nested": [{"value": '+value+'}]}')
                    with self.assertRaises(ValueError):adapter.bounded_json(path)

    def test_mid_read_real_file_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'changing.json';path.write_text('{}')
            real_fdopen=os.fdopen
            class MutatingReader:
                def __init__(self,*args,**kwargs):self.handle=real_fdopen(*args,**kwargs)
                def __enter__(self):return self
                def __exit__(self,*args):self.handle.close()
                def read(self,size):
                    result=self.handle.read(size)
                    path.write_text('{"changed":true}')
                    return result
            with patch.object(adapter.os,'fdopen',MutatingReader):
                with self.assertRaisesRegex(ValueError,'changed while reading'):
                    adapter.bounded_json(path)

    def test_cli_frozen_gold_and_exclusive_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);report,gold=evidence()
            (root/'r.json').write_text(json.dumps(report));(root/'g.json').write_text(json.dumps(gold))
            _,digest=adapter.bounded_json(root/'g.json')
            args=['--report',str(root/'r.json'),'--gold',str(root/'g.json'),'--gold-sha256',digest,
                  '--output',str(root/'p.json')]
            adapter.main(args)
            with self.assertRaises(FileExistsError):adapter.main(args)
            args[args.index('--gold-sha256')+1]='f'*64
            with self.assertRaises(ValueError):adapter.main(args)


if __name__=='__main__':unittest.main()
