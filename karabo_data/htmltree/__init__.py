from htmlgen import (Document, Element, Division, UnorderedList,
                     ListItem, html_attribute, Link, Script,
                     )
from pathlib import Path

_PKGDIR = Path(__file__).parent

class Style(Element):
    def __init__(self, styles):
        super().__init__('style')
        self.children.append_raw(styles)

class Summary(Element):
    def __init__(self, *content):
        super().__init__("summary")
        self.extend(content)

class Details(Element):
    def __init__(self, *content):
        super().__init__("details")
        self.extend(content)

def id_generator(template):
    n = 0
    while True:
        yield template % n
        n += 1

def make_list(items):
    ul = UnorderedList()
    ul.extend(items)
    return ul

def details_for_key(key, source_name):
    copylink = Link("#", "[📋]")
    copylink.set_attribute("data-copy-snippet",
                           ".get_array({!r}, {!r})".format(source_name, key))
    copylink.add_css_classes("karabodata-dataset-copylink")
    copylink.title = "Copy code snippet to get this data"
    return copylink

def data_collection_node(data):
    summary = "Collection of {} sources".format(len(data.all_sources))
    d = Details(Summary(summary), make_list(
        [source_node(s, data) for s in sorted(data.all_sources)]
    ))
    d.set_attribute("open", "")
    return d

def _nested_insert(nameparts, value, d: dict):
    head, *tail = nameparts
    if tail:
        return _nested_insert(tail, value, d.setdefault(head, {}))
    d[head] = value

def _format_keys(d: dict):
    for k, v in sorted(d.items()):
        if isinstance(v, dict):
            yield Details(Summary(k), make_list(_format_keys(v)))
        else:
            yield ListItem(k, ' ', v)

def source_node(source_name, data):
    children_d = {}
    for key in data._keys_for_source(source_name):
        nameparts = key.split('.')
        _nested_insert(nameparts, details_for_key(key, source_name), children_d)

    return Details(
        Summary(source_name),
        make_list(_format_keys(children_d))
    )

treeview_ids = id_generator("karabodata-treeview-container-%d")

def make_fragment(data):
    tree = data_collection_node(data)

    tv = Division(tree)
    tv.add_css_classes("karabodata-css-treeview")
    tv.id = next(treeview_ids)
    return tv

def get_treeview_css():
    with (_PKGDIR / 'treeview.css').open() as f:
        return Style(f.read())

JS_ACTIVATE_COPYLINKS_DOC = """
window.addEventListener("load", function(event) {
  enable_copylinks(document);
});
"""

JS_ACTIVATE_COPYLINKS_FRAG = """
enable_copylinks(document.getElementById("TREEVIEW-ID"));
"""

def get_copylinks_js(activation):
    with (_PKGDIR / "copysnippet.js").open() as f:
        return f.read().replace("//ACTIVATE", activation)

def make_document(obj):
    d = Document()
    d.append_head(get_treeview_css())
    d.append_head(Script(script=get_copylinks_js(JS_ACTIVATE_COPYLINKS_DOC)))
    d.title = "karabo_data"
    d.append_body(make_fragment(obj))
    return d

def h5obj_to_html(obj):
    treeview = make_fragment(obj)
    js_activate = JS_ACTIVATE_COPYLINKS_FRAG.replace("TREEVIEW-ID", treeview.id)

    div = Division(
        get_treeview_css(),
        treeview,
        Script(script=get_copylinks_js(js_activate)),
    )
    return str(div)

class DataTree:
    """View data sources in a Jupyter notebook"""
    def __init__(self, obj):
        self.obj = obj

    def _repr_html_(self):
        return h5obj_to_html(self.obj)

