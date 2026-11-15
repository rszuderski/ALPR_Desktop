"""Exercise the real MZ adapter with fake tensors/YOLO; no network/weights/GPU."""
from contextlib import nullcontext
from copy import deepcopy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from auto_annotation_tool.ranking import e2e_eval_runner as runner


class Tensor:
    def __init__(self,value):self.value=value
    def detach(self):return self
    def cpu(self):return self
    def tolist(self):return deepcopy(self.value)


@pytest.mark.parametrize('end2end',[True,False])
def test_real_adapter_receives_only_pixels_preserves_all_boxes_and_records_effective_parameters(tmp_path,monkeypatch,end2end):
    checkpoint=tmp_path/'fake.pt';checkpoint.write_bytes(b'SYNTHETIC CHECKPOINT - NO DESERIALIZATION')
    model_row={'checkpoint_file':str(checkpoint),'checkpoint_sha256':runner.mt._sha(checkpoint),'native_end2end':end2end}
    parameters=runner._inference('cpu',end2end)
    determinism={'deterministic_algorithms':True,'warn_only':True,'cudnn_benchmark':False,'cudnn_deterministic':True,
        'matmul_tf32':False,'cudnn_tf32':False,'cublas_workspace_config':':4096:8'}
    crop=np.zeros((100,320,3),np.uint8)
    boxes=SimpleNamespace(xyxy=Tensor([[1.,2.,10.,20.],[11.,2.,20.,20.]]),cls=Tensor([10.,35.]),conf=Tensor([.9,.8]))
    result=SimpleNamespace(orig_shape=(100,320),boxes=boxes,speed={'preprocess':.1,'inference':.2,'postprocess':.1})
    calls=[]
    model=SimpleNamespace(task='detect',names=deepcopy(runner.ALPHABET))
    model.names={int(k):v for k,v in model.names.items()}
    model.predictor=SimpleNamespace(args=SimpleNamespace(**parameters),imgsz=[320,320],
        model=SimpleNamespace(device='cpu',fp16=False,end2end=end2end,names=model.names))
    def predict(**kwargs):
        assert kwargs.pop('source') is crop
        assert kwargs.pop('stream') is False and kwargs==parameters
        assert kwargs['save'] is False and kwargs['save_crop'] is False and kwargs['half'] is False and kwargs['rect'] is False
        calls.append(1);return [result]
    model.predict=predict
    monkeypatch.setitem(sys.modules,'ultralytics',SimpleNamespace(YOLO=lambda *a,**k:model))
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(inference_mode=nullcontext,cuda=SimpleNamespace(is_available=lambda:False)))
    monkeypatch.setattr(runner.mt,'_actual_determinism',lambda:deepcopy(determinism))
    backend=runner.MZBackend(model_row,{'inference':parameters,'runtime':{'device':'cpu'}})
    try:
        row=backend.predict(crop)
        assert ''.join(r['character'] for r in row['characters'])=='AZ'
        assert [r['index'] for r in row['characters']]==[0,1] and len(calls)==1
        assert row['effective_inference']['backend_end2end'] is end2end
        boxes.conf=Tensor([.9])
        with pytest.raises(ValueError,match='ARRAY_LENGTH'):backend.predict(crop)
        boxes.conf=Tensor([.9,.8]);result.orig_shape=(320,100)
        with pytest.raises(ValueError,match='COORDINATE_SPACE'):backend.predict(crop)
    finally:backend.close()


def test_cpu_header_inspection_never_calls_predict_or_forward(tmp_path,monkeypatch):
    checkpoint=tmp_path/'fake.pt';checkpoint.write_bytes(b'NOT_REAL_WEIGHTS')
    def forbidden(*a,**k):pytest.fail('Header inspection must not run forward/predict')
    model=SimpleNamespace(task='detect',names={int(k):v for k,v in runner.ALPHABET.items()},predict=forbidden,
        model=SimpleNamespace(end2end=True,forward=forbidden,parameters=lambda:iter([SimpleNamespace(dtype='torch.float32')])))
    monkeypatch.setitem(sys.modules,'ultralytics',SimpleNamespace(YOLO=lambda *a,**k:model))
    info=runner.inspect_mz_checkpoint({'checkpoint_file':str(checkpoint),'checkpoint_sha256':runner.mt._sha(checkpoint)})
    assert info['names']==runner.ALPHABET and info['native_end2end'] is True and info['task']=='detect'
    assert info['loaded_parameter_dtype']=='torch.float32' and 'stored_parameter_dtype' not in info
