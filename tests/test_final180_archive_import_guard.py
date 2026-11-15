"""Three focused checks: original failure, bounded rejection, valid F6 import."""
from pathlib import Path
import ast,json,os,shutil,struct,time,tracemalloc,zipfile,zlib
import pytest
from auto_annotation_tool.ranking import mobile_package_experiments as packages
from auto_annotation_tool.ranking import final180_mobile as mobile,final180_timeline as timeline,final180_reference as ref,photo_reference as photo
from auto_annotation_tool.gui import final180_tv as tv
from test_final180_final_blockers import app_fixture,close_app


@pytest.fixture(scope='module')
def evidence():
    folder=os.environ.get('F6_IMPORT_EVIDENCE')
    if not folder:pytest.skip('Private real archive fixture not supplied')
    out=Path(folder);return out,photo.read_json(out/'paths.json')


def save(out,name,value):(out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),'utf-8')


def wait_tk(root,predicate,status):
    deadline=time.monotonic()+45
    while not predicate() and time.monotonic()<deadline:root.update();time.sleep(.01)
    assert predicate(),status()


def old_reader_method(out,name):
    tree=ast.parse((out/'before/mobile_package_experiments.py').read_text('utf-8-sig'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ReportBundleReader')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    namespace=dict(packages.__dict__)
    exec(compile(ast.Module(body=[method],type_ignores=[]),'before/mobile_package_experiments.py','exec'),namespace)
    return namespace[name]


def test_A_original_actual_button_failure(evidence,tmp_path,monkeypatch):
    out,paths=evidence;archive=Path(paths['private_archive'])
    assert archive.stat().st_size==374603776 and not zipfile.is_zipfile(archive)
    assert photo.digest(archive)=='7bf67f000a24d1baae5023b4eebf4974667ceed38f287e7168026e10a17593a5'
    monkeypatch.setattr(packages.ReportBundleReader,'read',old_reader_method(out,'read'))
    with pytest.raises(ValueError,match='374603776 B') as caught:packages.read_mobile_report_bundle(archive)
    assert 'JSON' in str(caught.value)
    root,app,errors=app_fixture(tmp_path,monkeypatch);dialogs=[]
    monkeypatch.setattr(tv.filedialog,'askopenfilename',lambda **kw:(dialogs.append(kw['title']) or str(archive)))
    try:
        app.import_btn.invoke()
        wait_tk(root,lambda:'374603776 B' in app.status.get(),app.status.get)
        assert 'Plik raportu JSON' in app.status.get() and len(dialogs)==1
        assert not app.result_windows and not app.manual_run_btn.winfo_manager() and not errors
        save(out,'A.json',dict(status='PASS_REPRODUCED_BEFORE_FIX',archive=str(archive),size=archive.stat().st_size,
            original_ui_error=app.status.get(),gate='ReportBundleReader._read_json_bundles',
            path=['FinalApp.import_android','final180_mobile.locate_run','read_mobile_report_bundle','ReportBundleReader.read','_read_json_report','_read_json_bundles'],
            analysis_started=False,result_window_opened=False,actual_format='TRUNCATED_ZIP_NOT_JSON'))
    finally:close_app(root,app)


def test_B_actual_corruption_clear_error_bounded_memory(evidence,tmp_path,monkeypatch):
    out,paths=evidence;archive=Path(paths['private_archive']);before=photo.digest(archive)
    with archive.open('rb') as stream:
        assert stream.read(4)==b'PK\x03\x04';stream.seek(-8*1024*1024,2);tail=stream.read()
    assert b'PK\x05\x06' not in tail and b'PK\x01\x02' not in tail
    position=tail.rfind(b'PK\x03\x04');header=struct.unpack_from('<4s5H3I2H',tail,position);n,e=header[-2:]
    name=tail[position+30:position+30+n].decode('utf-8');decoder=zlib.decompressobj(-15)
    raw=decoder.decompress(tail[position+30+n+e:]);assert not decoder.eof and raw.startswith(b'\xff\xd8') and not raw.endswith(b'\xff\xd9')
    del tail,raw
    def forbid_json(*a,**kw):raise AssertionError('Binary archive entered JSON reader')
    monkeypatch.setattr(packages.ReportBundleReader,'_read_json_bundles',forbid_json)
    tracemalloc.start()
    for read in (packages.read_mobile_report_bundle,packages.read_mobile_report_bundles):
        with pytest.raises(ValueError,match='Import integralności: archiwum ZIP') as caught:read(archive)
        assert 'JSON' not in str(caught.value)
    _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    assert peak<2*1024*1024,peak
    # A broken ZIP renamed .json is still detected by its signature.
    renamed=tmp_path/'truncated.json';renamed.write_bytes(b'PK\x03\x04'+b'\0'*64)
    with pytest.raises(ValueError,match='archiwum ZIP'):packages.read_mobile_report_bundle(renamed)
    empty=tmp_path/'empty.alprsession';empty.write_bytes(b'')
    with pytest.raises(ValueError,match='archiwum ZIP'):packages.read_mobile_report_bundle(empty)
    root,app,errors=app_fixture(tmp_path,monkeypatch);dialogs=[]
    monkeypatch.setattr(tv.filedialog,'askopenfilename',lambda **kw:(dialogs.append(kw['title']) or str(archive)))
    try:
        app.import_btn.invoke();wait_tk(root,lambda:'Import integralności:' in app.status.get(),app.status.get)
        assert 'JSON' not in app.status.get() and len(dialogs)==1 and not app.result_windows and not errors
        assert not app.manual_run_btn.winfo_manager()  # picking a run cannot fix a corrupt archive
        assert photo.digest(archive)==before
        # PC evidence can be checked independently; it cannot repair Android evidence.
        prepared=Path(paths['full_prepared']);m=photo.read_json(prepared/'manifest.json')
        state,events,_=timeline.read_run(paths['full_run'],m,photo.digest(prepared/'manifest.json'))
        pc=timeline.project(m,events);gt=ref.reference_preflight(prepared)
        assert len(pc)==176 and all(r['status']=='COMPLETED' for r in pc)
        assert gt['counts']['GT_COMPLETE']==176 and gt['complete_gt_plates']==209
        save(out,'B.json',dict(status='PASS_BOUNDED_INTEGRITY_REJECTION',final176_analysis='BLOCKED_INCOMPLETE_ZIP',
            ui_error=app.status.get(),archive_sha256=before,last_entry=name,last_deflate_complete=False,source_unchanged=True,
            peak_python_allocated_bytes=peak,source_size_bytes=archive.stat().st_size,dialogs=1,
            import_integrity='FAIL',run_association='NOT_ATTEMPTED',assessability='NOT_AVAILABLE',result_preview='NOT_CREATED',
            pc_completed=176,gt_photographs=176,gt_plates=209,android_session_state='UNVERIFIED_FROM_TRUNCATED_EXPORT',
            pc_run_id=state['run_id'],manifest_sha256=photo.digest(prepared/'manifest.json')))
    finally:close_app(root,app)


def test_C_small_f2_real_import_partial_na_and_first_mt(evidence,tmp_path,monkeypatch):
    out,paths=evidence;small=Path(paths['small_evidence']);old=photo.read_json(small/'private/baseline.json')
    source_paths=photo.read_json(small/'paths.json');source=Path(source_paths['private_archive']);before=photo.digest(source)
    archive=out/'private/small_f2.alprsession';shutil.copy2(source,archive)
    bundle=packages.read_mobile_report_bundle(archive)
    assert bundle.validation.ok and bundle.f6_evidence['status']=='PARTIAL'
    assert bundle.collection_session['state']=='PARTIAL'
    # Ordinary JSON and old ZIP remain accepted; the real JSON limit is intact.
    report=tmp_path/'legacy.json';report.write_text(json.dumps(dict(schema=packages.MOBILE_BENCHMARK_REPORT_SCHEMA,report_id='synthetic')),'utf-8')
    assert packages.read_mobile_report_bundle(report).validation.ok
    many=tmp_path/'many.json';many.write_text(json.dumps([json.loads(report.read_text())]),'utf-8')
    assert len(packages.read_mobile_report_bundles(many))==1
    legacy=tmp_path/'legacy.zip'
    with zipfile.ZipFile(legacy,'w',zipfile.ZIP_DEFLATED) as z:z.write(report,'report.json')
    assert packages.read_mobile_report_bundle(legacy).validation.ok
    huge=tmp_path/'huge.json'
    with huge.open('wb') as stream:stream.write(b'{');stream.truncate(packages.MOBILE_REPORT_MAX_TEXT_BYTES+1)
    with pytest.raises(ValueError,match='Plik raportu JSON jest za duży'):packages.read_mobile_report_bundle(huge)
    assert packages.MOBILE_REPORT_MAX_TEXT_BYTES==32*1024*1024
    root,app,errors=app_fixture(tmp_path,monkeypatch);dialogs=[];panels=[];real_window=tv.FinalResultsWindow;analyze=mobile.analyze_import
    def private_export(a,r,**kwargs):return analyze(a,r,output=out/'small_regression_results',**kwargs)
    def hidden_window(*a,**kwargs):
        panel=real_window(*a,**kwargs);panel.window.withdraw();panels.append(panel);return panel
    monkeypatch.setattr(mobile,'analyze_import',private_export)
    monkeypatch.setattr(tv,'FinalResultsWindow',hidden_window)
    monkeypatch.setattr(tv.filedialog,'askopenfilename',lambda **kw:(dialogs.append(kw['title']) or str(archive)))
    try:
        app.import_btn.invoke();wait_tk(root,lambda:bool(panels),app.status.get)
        panel=panels[0];result=panel.report
        assert len(dialogs)==1 and panel.window.title()=='Wyniki pokazu i ocena względem GT' and len(panel.rows)==5
        for key in ('summary','rows','frames','raw_mt','raw_mz','presentation_timeline','integration_gaps'):assert result[key]==old[key],key
        assert result['summary']['raw_mt_calls']==11 and result['summary']['selected_mt_exposures']==0
        assert result['metrics_status']=='NOT_AVAILABLE_NO_GT' and 'EARLIER_MT_INPUT_MAY_BE_UNLOCATED' in result['integration_gaps']
        for eid in panel.rows:
            panel.tree.selection_set(eid);panel.select();root.update()
        assert not errors and photo.digest(source)==photo.digest(archive)==before
        save(out,'C.json',dict(status='PASS_REAL_F2_SINGLE_FILE_NATIVE_TK',source_sha256=before,result_path=str(panel.path),
            import_integrity='PASS',run_association='PASS_TWO_QR_FOUR_ARUCO',android_state='PARTIAL',
            assessability='N/A_NO_GT_AND_UNLOCATED_FIRST_MT',result_preview='PASS_NATIVE_TK',dialog_count=1,
            planned=5,raw_mt=11,raw_mz=2,qualifying_first_mt=0,all_rows_and_evidence_unchanged=True,real_json_limit_unchanged=True))
        panel.window.destroy()
    finally:close_app(root,app)
