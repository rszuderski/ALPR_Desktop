import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import xml.etree.ElementTree as ET

from PIL import Image
import pytest

from auto_annotation_tool.gui import z3_extraction_sources as sources
from auto_annotation_tool.gui import z3_extraction_tab_ui as extraction
from auto_annotation_tool.gui import z3_preview_records as records
from auto_annotation_tool.gui.z3_approved_source import build_approved_plate_source, bind_approved_source


class Value:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


@pytest.fixture
def data(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    image = images / "BI360FE.jpg"
    Image.new("RGB", (100, 100), (90, 100, 110)).save(image)
    raw = tmp_path / "raw" / "annotations.xml"
    raw.parent.mkdir()
    raw.write_text('''<annotations><image name="BI360FE.jpg" width="100" height="100">
      <polygon label="plate" points="5,5;95,5;95,95;5,95"><attribute name="plate_annotation_id">wrong-car</attribute></polygon>
      <polygon label="plate" points="30,60;65,60;65,70;30,70"><attribute name="plate_annotation_id">correct-plate</attribute></polygon>
      </image><image name="unapproved.jpg" width="100" height="100"><polygon label="plate" points="1,1;20,1;20,5;1,5"/></image></annotations>''', encoding="utf-8")
    entries = [{"image_name":image.name,"source_image_path":str(image),"width":100,"height":100,
                "plates":[{"polygon":[[30,60],[65,60],[65,70],[30,70]],"confidence":1.,
                           "attributes":{"plate_annotation_id":"correct-plate","ground_truth_text":"BI360FE",
                                         "manually_edited":"true","manual_source":"preview"}}]}]
    return tmp_path, images, raw, entries


def host_for(raw, images):
    return SimpleNamespace(annotation_run_dir_var=Value(str(raw.parent)), xml_path_var=Value(str(raw)),
                           images_dir_var=Value(str(images)), _paths_equivalent=lambda a,b:Path(a).resolve()==Path(b).resolve())


def test_approved_geometry_is_used_instead_of_all_polygons_on_an_ok_image(data):
    root, images, raw, entries = data
    before = raw.read_bytes()
    result = build_approved_plate_source(entries, root / "state")
    tree = ET.parse(result["xml_path"])
    polygons = tree.findall('.//polygon')
    assert len(tree.findall('.//image')) == len(polygons) == 1
    assert polygons[0].findtext("attribute[@name='plate_annotation_id']") == "correct-plate"
    assert polygons[0].findtext("attribute[@name='ground_truth_text']") == "BI360FE"
    assert result["images_dir"] == images
    assert raw.read_bytes() == before
    assert not list(result["run_dir"].rglob('*.jpg')), "No image copies during entry"
    before_stat = result["xml_path"].stat().st_mtime_ns
    assert build_approved_plate_source(entries,root/"state") == result
    assert result["xml_path"].stat().st_mtime_ns == before_stat


def test_existing_wrong_crop_set_is_stale_and_gt_only_changes_keep_geometry_current(data):
    root, images, raw, entries = data
    host = host_for(raw, images)
    stale = {"plate_count":3,"source":sources.current_extract_source_payload_with_signature(host)}
    current = build_approved_plate_source(entries,root/"state")
    bind_approved_source(host,current)
    assert sources.extract_manifest_freshness(host,stale)["recrop_required"]
    good = {"plate_count":1,"source":sources.current_extract_source_payload_with_signature(host)}
    assert sources.extract_manifest_matches_current_source(host,good)
    changed = copy.deepcopy(entries)
    changed[0]["plates"][0]["attributes"]["ground_truth_text"] = "NEW123"
    bind_approved_source(host,build_approved_plate_source(changed,root/"state"))
    freshness = sources.extract_manifest_freshness(host,good)
    assert freshness["geometry_current"] and freshness["gt_metadata_stale"]
    assert not freshness["recrop_required"]


def test_empty_approved_pool_never_falls_back_to_raw_xml(data):
    root, images, raw, entries = data
    with pytest.raises(ValueError,match="Brak zatwierdzonych"):
        build_approved_plate_source([],root/"state")


def add_vehicle_conflict_evidence(root, raw):
    state = root / "_campaign_state"; state.mkdir()
    tree = ET.parse(raw)
    first = tree.getroot().find('image')
    ET.SubElement(first, 'box', label='vehicle', xtl='5', ytl='5', xbr='95', ybr='95')
    tree.write(raw,encoding='utf-8',xml_declaration=True)
    (state/'project.json').write_text(json.dumps({'project':{'project_start_plate_source_xml':str(raw)}}),encoding='utf-8')
    return state


def test_image_ok_does_not_admit_a_proven_vehicle_polygon_into_crops(data):
    root, images, raw, entries = data
    state = add_vehicle_conflict_evidence(root,raw)
    entries[0]['plates'].insert(0,{'polygon':[[5,5],[95,5],[95,95],[5,95]],'confidence':.9,
                                  'attributes':{'plate_annotation_id':'wrong-car','annotation_origin':'auto'}})
    before = copy.deepcopy(entries)
    result = build_approved_plate_source(entries,state)
    assert result['approved_plates'] == 1
    assert ET.parse(result['xml_path']).findtext(".//attribute[@name='plate_annotation_id']") == 'correct-plate'
    assert entries == before


def test_resynchronizing_old_xml_does_not_reinsert_vehicle_plate_into_approved_set(data, monkeypatch):
    from auto_annotation_tool.gui import z2_campaign_runtime as runtime, z2_preview_workflow as parser
    root, images, raw, entries = data
    add_vehicle_conflict_evidence(root,raw)
    monkeypatch.setattr(runtime,'_resolve_campaign_project_name_from_run_dir',lambda p:'')
    monkeypatch.setattr(runtime,'CAMPAIGN',SimpleNamespace(get_current_iteration_num=lambda:1))
    host = SimpleNamespace(
        _is_free_mode_session_context=lambda:False,
        _resolve_safe_annotation_run_dir=lambda p,**kw:Path(p),
        _bbox_from_polygon=lambda p:(min(x for x,y in p),min(y for x,y in p),max(x for x,y in p),max(y for x,y in p)),
        _keypoints_from_polygon=lambda p:[(x,y,1.) for x,y in p],
        _load_annotation_run_manifest=lambda p:{'approved_filenames':['bi360fe.jpg'],'input_dir':str(images)},
        _load_annotation_run_approved_filenames=lambda p:{'bi360fe.jpg'},
        _resolve_existing_dir=lambda p:Path(p) if p and Path(p).is_dir() else None,
        current_input_dir=images, _annotation_run_manifest_has_manual_value=lambda m:True,
        _get_plate_detections=lambda a:a.plates,
        _build_campaign_plate_entry_key=lambda **kw:str(kw['source_image_path']),
        _detection_polygon=lambda d:d.polygon,
    )
    host._parse_cvat_preview_annotations=lambda p:parser._parse_cvat_preview_annotations(host,p)
    result=runtime._build_campaign_plate_approved_entries_from_run(host,raw.parent,force_parse_xml=True)
    assert len(result)==1 and result[0]['plate_count']==1
    assert result[0]['plates'][0]['attributes']['plate_annotation_id']=='correct-plate'


def test_campaign_rebinds_approved_source_before_considering_old_cached_crops(data, monkeypatch):
    root, images, raw, entries = data
    host = host_for(raw,images)
    campaign = SimpleNamespace(get_project_state_dir=lambda:root/"state",list_plate_approved_entries=lambda:entries)
    monkeypatch.setattr(extraction,"CAMPAIGN",campaign)
    monkeypatch.setattr(extraction,"_campaign_step3_context_active",lambda h:True)
    class Inspected(BaseException):
        pass
    def ready():
        assert Path(host.xml_path_var.get()) != raw
        assert sources.current_extract_source_signature(host)["xml_plate_count"] == 1
        raise Inspected
    host._is_extract_preview_ready = ready
    with pytest.raises(Inspected):
        extraction.run_extraction(host)


def test_t05_entry_ignores_ready_but_unfiltered_xml_before_preview_reuse(data, monkeypatch):
    from auto_annotation_tool.gui import z3_campaign_flow as flow
    root, images, raw, entries = data
    host = host_for(raw,images)
    campaign = SimpleNamespace(get_active_project_name=lambda:'test',get_current_step=lambda:3,
        get_dir=lambda kind:root,get_iteration_target=lambda:'char',get_current_iteration_num=lambda:1,
        get_iteration_image_source_dir=lambda iteration:images,get_project_state_dir=lambda:root/'state',
        list_plate_approved_entries=lambda:entries)
    monkeypatch.setattr(flow,'CAMPAIGN',campaign)
    class EntryChecked(BaseException):
        pass
    def check(xml, folder):
        assert Path(xml) != raw
        assert len(ET.parse(xml).findall('.//polygon')) == 1
        assert Path(folder) == images
        raise EntryChecked
    monkeypatch.setattr(flow,'_campaign_preview_entry_key',check)
    with pytest.raises(EntryChecked):
        flow.open_campaign_step3_entry(host,{'restore_run_dir':str(raw.parent),'xml_path':str(raw),'input_dir':str(images)})


def test_z2_direct_handoff_uses_images_root_of_approved_snapshot(data):
    from auto_annotation_tool.gui import z2_miniflow_runtime as flow
    root, images, raw, entries = data
    source = build_approved_plate_source(entries,root/'state')
    character_tab = SimpleNamespace(set_pending_z2_annotation_source=Mock())
    class HandoffChecked(BaseException):
        pass
    def open_tab(_name):
        assert character_tab.set_pending_z2_annotation_source.call_args.kwargs['images_dir'] == str(images)
        raise HandoffChecked
    host = SimpleNamespace(is_processing=False,_get_preferred_annotation_run_dir=lambda **kw:raw.parent,
        _ensure_preview_edits_saved=lambda reason:True,_resolve_step3_images_dir_from_z2_run=lambda run:root,
        _prepare_approved_step3_source_from_z2_run=lambda *a:source['run_dir'],
        _load_annotation_run_manifest=lambda run:{'input_dir':str(images)},
        _resolve_existing_dir=lambda p:Path(p),app=SimpleNamespace(tabs={'characters':character_tab},open_controlled_tab=open_tab))
    with pytest.raises(HandoffChecked):
        flow._open_step3_from_z2_annotation_source(host)


def test_real_crop_worker_uses_only_approved_geometry_and_gt(data, monkeypatch):
    root, images, raw, entries = data
    out = root / "crops"; out.mkdir()
    host = host_for(raw,images)
    host.preview_dir_var = Value()
    host._project_reset_token = 1
    host._step3_linear_mode = True
    host.interpolation_var = Value("lanczos4")
    host.frame = SimpleNamespace(after=lambda delay,callback:callback())
    host.app = SimpleNamespace()
    host.ext_log = None
    for name in ("btn_extract","btn_ext_stop","ext_progress"):
        setattr(host,name,Mock())
    for name in ("_force_save_all","_reset_preview_cache","_refresh_extract_workflow_ui","_set_extraction_status",
                 "_log","_refresh_extract_action_state","_show_campaign_extract_failure_modal"):
        setattr(host,name,Mock())
    host._is_extract_preview_ready = lambda:False
    host._refresh_source_binding_status = lambda **kw:{"ok":True}
    host._get_step3_chars_root_dir = lambda **kw:out
    host._get_campaign_step3_min_extracted_plate_count = lambda:1
    host._plate_cut_reading_order_key = sources.plate_cut_reading_order_key
    host._prepare_plate_cut_detections_for_source = lambda name,plates:sources.prepare_plate_cut_detections_for_source(host,name,plates)
    host._atomic_write_json = lambda path,obj:Path(path).write_text(json.dumps(obj),encoding="utf-8")
    host._write_extract_source_manifest = lambda run_dir,**kw:sources.write_extract_source_manifest(host,run_dir,**kw)
    seed = {"source_image":str(images/"BI360FE.jpg"),"source_plate_index":1,"source_bbox":[30,60,65,70],
            "characters":[{"character":"B","bbox":[1,2,10,20],"method":"manual"}],"status":"edited"}
    wrong_seed = {"source_image":str(images/"BI360FE.jpg"),"source_plate_index":0,"source_bbox":[5,5,95,95],
                  "characters":[{"character":"X","bbox":[1,1,20,20]}]}
    host._campaign_step3_reextract_seed_metadata = {
        key:item for item in (seed,wrong_seed) for key in records.build_preview_plate_reextract_match_keys(item)}
    host._load_json_file_safely = lambda p:json.loads(Path(p).read_text(encoding="utf-8"))
    host._ensure_plate_source_metadata = Mock()
    host._get_preview_plate_image_size = lambda directory,pid:Image.open(Path(directory)/"images"/f"{pid}.jpg").size
    host._recalculate_preview_statuses_in_metadata = lambda data:data
    host._merge_reextract_seed_metadata_into_preview = lambda run_dir,**kw:records.merge_reextract_seed_metadata_into_preview(host,run_dir,**kw)
    campaign = SimpleNamespace(get_project_state_dir=lambda:root/"state",list_plate_approved_entries=lambda:entries,
                               get_active_project_name=lambda:"test",get_iteration_target=lambda:"char",set_step3_preview_dir=Mock())
    monkeypatch.setattr(extraction,"CAMPAIGN",campaign)
    monkeypatch.setattr(extraction,"_campaign_step3_context_active",lambda h:True)
    monkeypatch.setattr(extraction,"collect_image_resource_gt_pack_paths",lambda *a,**kw:[])
    monkeypatch.setattr(extraction,"ExtractionProgress",lambda *a:SimpleNamespace(publish=Mock(),close=Mock()))
    monkeypatch.setattr(extraction,"_continue_campaign_pz1_to_pz2",Mock())
    monkeypatch.setattr(extraction.threading,"Thread",lambda target,**kw:SimpleNamespace(start=target))
    extraction.run_extraction(host)
    host._show_campaign_extract_failure_modal.assert_not_called()
    metadata = json.loads((Path(host.preview_dir_var.get())/'metadata.json').read_text(encoding='utf-8'))
    assert len(metadata) == 1
    item = next(iter(metadata.values()))
    assert item['source_annotation_id'] == 'correct-plate'
    assert item['source_bbox'] == [30.,60.,65.,70.]
    assert item['ground_truth_text'] == 'BI360FE'
    assert item['characters'] == seed['characters']
    manifest = json.loads((Path(host.preview_dir_var.get())/'extract_manifest.json').read_text(encoding='utf-8'))
    assert sources.extract_manifest_matches_current_source(host,manifest)
