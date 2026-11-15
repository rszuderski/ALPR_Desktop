"""Visible viewport plus individual reachability in the existing sidebar scroll."""
def assert_sidebar_reaches(root,host,widgets):
    scroll=host.review_sidebar_scroll
    assert scroll.winfo_ismapped()
    saved=scroll.yview()[0]
    root.update_idletasks()
    top=scroll.winfo_rooty();bottom=top+scroll.winfo_height()
    assert top>=host.window.winfo_rooty()
    assert bottom<=host.window.winfo_rooty()+host.window.winfo_height()
    total=scroll.bbox('all')[3]
    for widget in widgets:
        assert widget.winfo_ismapped()
        y=widget.winfo_rooty()-top+scroll.canvasy(0)
        scroll.yview_moveto(max(0,(y-(scroll.winfo_height()-widget.winfo_height())/2)/total))
        root.update_idletasks()
        assert top-1<=widget.winfo_rooty()
        assert widget.winfo_rooty()+widget.winfo_height()<=bottom+1
        assert widget.winfo_rootx()>=scroll.winfo_rootx()
        assert widget.winfo_rootx()+widget.winfo_width()<=scroll.winfo_rootx()+scroll.winfo_width()
    scroll.yview_moveto(saved)
    root.update_idletasks()
