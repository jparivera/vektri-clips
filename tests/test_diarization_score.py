"""Pure standard-library scorer tests: no inference, downloads or third-party media."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import wave

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import score_diarization as der


def segment(start, end, speaker):
    return {'start': start, 'end': end, 'speaker': speaker}


class ScoringTests(unittest.TestCase):
    def test_perfect_swapped_speakers(self):
        ref=[segment(0,2,'A'),segment(3,5,'B')]
        hyp=[segment(0,2,'label9'),segment(3,5,'label3')]
        for result in der.evaluate(ref,hyp,6):
            self.assertEqual(result['der'],0)
            self.assertEqual(result['hypothesis_to_reference_mapping'],{'label3':'B','label9':'A'})

    def test_excess_speaker_is_false_alarm_and_unmapped(self):
        result=der.score_condition([segment(0,2,'A')],[segment(0,2,'x'),segment(0,2,'y')],2)
        self.assertEqual(result['reference_speaker_seconds'],2)
        self.assertEqual(result['false_alarm_seconds'],2)
        self.assertEqual(result['speaker_confusion_seconds'],0)
        self.assertEqual(list(result['hypothesis_to_reference_mapping'].values()).count(None),1)
        self.assertEqual(result['der'],1)

    def test_overlap_missing_second_voice(self):
        ref=[segment(0,2,'A'),segment(1,3,'B')]
        hyp=[segment(0,2,'x'),segment(2,3,'y')]
        included=der.score_condition(ref,hyp,3)
        excluded=der.score_condition(ref,hyp,3,exclude_reference_overlap=True)
        self.assertEqual(included['reference_speaker_seconds'],4)
        self.assertEqual(included['missed_speech_seconds'],1)
        self.assertEqual(included['der'],.25)
        self.assertEqual(excluded['reference_speaker_seconds'],2)
        self.assertEqual(excluded['der'],0)
        self.assertEqual(excluded['reference_overlap_excluded_seconds'],1)

    def test_short_turn_removed_by_collar_and_silence_false_alarm_retained(self):
        result=der.score_condition([segment(0,.4,'A')],[segment(0,2,'x')],2,.25)
        self.assertAlmostEqual(result['collar_excluded_seconds'],.65)
        self.assertAlmostEqual(result['false_alarm_seconds'],1.35)
        self.assertEqual(result['reference_speaker_seconds'],0)
        self.assertIsNone(result['der'])

    def test_silence_false_alarm_and_zero_denominator(self):
        result=der.score_condition([],[segment(.5,1.5,'x')],2)
        self.assertEqual(result['false_alarm_seconds'],1)
        self.assertEqual(result['hypothesis_to_reference_mapping'],{'x':None})
        self.assertTrue(result['denominator_zero'])
        self.assertIsNone(result['der_percent'])

    def test_silence_perfect_has_null_der(self):
        result=der.score_condition([],[],2)
        self.assertEqual(result['error_speaker_seconds'],0)
        self.assertIsNone(result['der'])

    def test_complete_miss(self):
        result=der.score_condition([segment(.5,1.5,'A')],[],2)
        self.assertEqual(result['missed_speech_seconds'],1)
        self.assertEqual(result['der'],1)

    def test_der_can_exceed_one(self):
        result=der.score_condition([segment(0,1,'A')],[segment(0,4,'x')],4)
        self.assertEqual(result['false_alarm_seconds'],3)
        self.assertEqual(result['der'],3)

    def test_crossing_same_speaker_segments_not_double_counted(self):
        result=der.score_condition([segment(0,2,'A'),segment(1,3,'A')],
                                   [segment(0,1.5,'x'),segment(1.5,3,'y')],3)
        self.assertEqual(result['reference_speaker_seconds'],3)
        self.assertEqual(result['reference_overlap_total_seconds'],0)
        self.assertEqual(result['speaker_confusion_seconds'],1.5)
        self.assertEqual(result['der'],.5)

    def test_duplicate_hypothesis_same_speaker_not_false_alarm(self):
        result=der.score_condition([segment(0,3,'A')],[segment(0,2,'x'),segment(1,3,'x')],3)
        self.assertEqual(result['hypothesis_speaker_seconds'],3)
        self.assertEqual(result['der'],0)

    def test_mapping_excludes_masked_overlap_before_optimization(self):
        ref=[segment(0,10,'A'),segment(0,9,'B')]
        hyp=[segment(0,9,'x'),segment(9,10,'y')]
        included=der.score_condition(ref,hyp,10)
        excluded=der.score_condition(ref,hyp,10,exclude_reference_overlap=True)
        self.assertEqual(included['hypothesis_to_reference_mapping'],{'x':'B','y':'A'})
        self.assertEqual(excluded['hypothesis_to_reference_mapping'],{'x':None,'y':'A'})
        self.assertEqual(excluded['der'],0)

    def test_unquantized_boundaries(self):
        result=der.score_condition([segment(.123,.987,'A')],[segment(.128,.982,'x')],1)
        self.assertAlmostEqual(result['reference_speaker_seconds'],.864)
        self.assertAlmostEqual(result['missed_speech_seconds'],.01)

    def test_collar_includes_internal_input_reference_boundaries(self):
        result=der.score_condition([segment(0,2,'A'),segment(1,3,'A')],[],3,.25)
        self.assertAlmostEqual(result['collar_excluded_seconds'],1.5)
        self.assertAlmostEqual(result['reference_speaker_seconds'],1.5)

    def test_invalid_inputs(self):
        bad_segments=[None,{},[None],[segment(-1,1,'A')],[segment(1,1,'A')],
                      [segment(0,3,'A')],[segment(False,1,'A')],[segment(0,float('nan'),'A')],
                      [segment(0,1,'')],[segment(0,1,7)]]
        for bad in bad_segments:
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                der.score_condition(bad,[],2)
        for duration in (0,-1,True,float('inf')):
            with self.subTest(duration=duration),self.assertRaises(ValueError):
                der.score_condition([],[],duration)
        with self.assertRaises(ValueError):
            der.score_condition([segment(0,1,str(i)) for i in range(7)],[],2)


class InputIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.audio=self.root/'fixture.wav'
        self.pcm=b'\0\0'*16000
        with wave.open(str(self.audio),'wb') as wav:
            wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(16000);wav.writeframes(self.pcm)
        self.gold={'schema_version':1,'duration_seconds':1,
                   'reference_type':'constructed_utterance_ownership',
                   'limitations':['No human VAD; internal pauses not separately annotated.'],
                   'audio':{'path':'fixture.wav','sha256':hashlib.sha256(self.audio.read_bytes()).hexdigest(),
                            'pcm_sha256':hashlib.sha256(self.pcm).hexdigest()},'segments':[]}
        self.gold_path=self.root/'gold.json'
        self.gold_path.write_text(json.dumps(self.gold))
        self.gold_hash=hashlib.sha256(self.gold_path.read_bytes()).hexdigest()
        self.pred=self.root/'prediction.json';self.pred.write_text(json.dumps({'segments':[]}))

    def tearDown(self):
        self.temp.cleanup()

    def arguments(self):
        return ['--gold',str(self.gold_path),'--gold-sha256',self.gold_hash,
                '--prediction',str(self.pred),'--output',str(self.root/'report.json')]

    def test_cli_verifies_audio_pcm_and_preserves_reference_scope(self):
        result=der.main(self.arguments())
        self.assertEqual(result['audio']['sample_frames'],16000)
        self.assertEqual(result['reference_type'],self.gold['reference_type'])
        self.assertEqual(result['reference_limitations'],self.gold['limitations'])
        self.assertEqual(len(result['conditions']),4)
        self.assertFalse(result['model_inference_performed'])

    def test_changed_gold_is_rejected(self):
        self.gold_path.write_text(self.gold_path.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'Gold SHA'):
            der.main(self.arguments())

    def test_changed_audio_is_rejected(self):
        self.audio.write_bytes(self.audio.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'Audio file SHA'):
            der.main(self.arguments())

    def test_wrong_pcm_hash_rejected(self):
        self.gold['audio']['pcm_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'PCM SHA'):
            der.verify_audio(self.gold_path,self.gold)

    def test_escape_path_rejected(self):
        self.gold['audio']['path']='../fixture.wav'
        with self.assertRaisesRegex(ValueError,'contained relative'):
            der.verify_audio(self.gold_path,self.gold)

    def test_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as other:
            outside=Path(other)/'audio.wav';outside.write_bytes(self.audio.read_bytes())
            (self.root/'escape.wav').symlink_to(outside)
            self.gold['audio']['path']='escape.wav'
            with self.assertRaisesRegex(ValueError,'symlink escapes'):
                der.verify_audio(self.gold_path,self.gold)

    def test_output_not_overwritten(self):
        (self.root/'report.json').write_text('preserve')
        with self.assertRaises(FileExistsError):
            der.main(self.arguments())
        self.assertEqual((self.root/'report.json').read_text(),'preserve')

    def test_wrong_declared_prediction_source_rejected(self):
        self.pred.write_text(json.dumps({'segments':[],'source_sha256':'0'*64}))
        with self.assertRaisesRegex(ValueError,'different audio'):
            der.main(self.arguments())


if __name__=='__main__':
    unittest.main()
