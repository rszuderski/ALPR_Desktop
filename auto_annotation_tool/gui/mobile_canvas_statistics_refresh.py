"""Non-blocking statistics from one saved review, without reloading Canvas."""
from __future__ import annotations
import copy
from ..ranking.mobile_session_metrics import current,result_key


def cancel(host):
    timer=getattr(host,'_canvas_statistics_after',None)
    if timer:
        try:host.window.after_cancel(timer)
        except Exception:pass  # The toplevel may already have been destroyed.
    host._canvas_statistics_after=None
    host._canvas_statistics_pending=None


def request(host,*,after):
    if host._closed or host._saving or getattr(host,'_canvas_statistics_job',None) is not None:
        return False
    owner=host.session
    # _save_change always replaces the saved review (including Canvas's
    # copy-on-write annotations). Pin that immutable version, not the mutable
    # owner whose review pointer can advance during the calculation.
    snapshot=copy.copy(owner)
    review=snapshot.review
    token=object();host._canvas_statistics_job=token

    def fresh():return not host._closed and host.session is owner and owner.review is review

    def work():
        try:
            from .z4_mobile_sample_review import (_pose_geometry_rows,_pose_geometry_summary,
                make_scientific_summary,verification_progress,MobilePackageExperimentStore)
            stats=snapshot.statistics()
            if not fresh():return stats,None,'',None,None
            geometry=_pose_geometry_rows(snapshot)
            prepared=dict(geometry=geometry,science=make_scientific_summary(snapshot,stats,geometry),
                form=verification_progress(snapshot),pose=_pose_geometry_summary(snapshot))
            warning='';updated_report=None
            if fresh():
                try:
                    store=MobilePackageExperimentStore(host.browser.store.root_dir)
                    if store.apply_human_review(snapshot):
                        updated_report=next((r for r in store.reports if r.report_id==host.report.report_id
                            and r.source_archive_sha256==review.source_archive_sha256),None)
                except Exception as exc:warning='Decyzje zapisane; ranking wymaga odświeżenia: '+str(exc)
            return stats,prepared,warning,updated_report,None
        except Exception as exc:return None,None,'',None,str(exc)

    def publish():
        host._canvas_statistics_after=None
        pending=getattr(host,'_canvas_statistics_pending',None)
        if host._closed or pending is None:return
        if pending[0] is not token:return
        if not fresh():
            host._canvas_statistics_pending=None
            host._canvas_editor.schedule_refresh()
            return
        ed=host._canvas_editor
        if (host._saving or ed.busy or ed.drag or ed.points or ed.entry is not None
                or ed.mode=='draw' or getattr(host,'_view_drag_at',None)):
            host._canvas_statistics_after=host.window.after(40,publish)
            return
        _,stats,prepared,warning,updated_report=pending
        host._canvas_statistics_pending=None
        if not current(stats.get('live_session_metrics_v2'),result_key(owner)):
            ed.schedule_refresh();return
        # Publish values and rows only. _changed/_render_subject would reload
        # the photograph and could replace a newer geometry selection.
        host.stats=stats
        host._refresh_subject_rows(prepared)
        host._refresh_stats(prepared)
        ed.sync();ed.update_report();ed.request_overlay()
        host._publish_review_report(updated_report)
        if warning:host.status_var.set(warning)

    def done(result):
        if getattr(host,'_canvas_statistics_job',None) is token:host._canvas_statistics_job=None
        if host._closed:return
        after()
        stats,prepared,warning,updated_report,error=result
        if not fresh():
            host._canvas_editor.schedule_refresh();return
        if error:
            host.status_var.set('Nie odświeżono wyników: '+error)
            return
        cancel(host)
        host._canvas_statistics_pending=(token,stats,prepared,warning,updated_report)
        publish()

    host._submit(work,done,saving=False,refresh_controls=False)
    return True
