# import anywidget
import time
import traitlets
import json
import ipyreact
import pathlib
from IPython.display import HTML, display
from IPython import get_ipython

ipyreact.define_import_map({
  "d3-scale-chromatic": "https://esm.sh/d3-scale-chromatic@3.1.0"
})

css_id = "topk-widget-css"
class TopK(ipyreact.Widget):
    data = traitlets.Dict({}).tag(sync=True)
    features = traitlets.List([]).tag(sync=True)
    n = traitlets.Int(10).tag(sync=True)

    @traitlets.validate('data')
    def _validate_data(self, proposal):
        return self._ensure_serializable(proposal['value'])
    
    def _ensure_serializable(self, data):
        def convert(item):
            if hasattr(item, 'tolist'):  # Check if it's a tensor-like object
                return item.tolist()
            elif isinstance(item, dict):
                return {k: convert(v) for k, v in item.items()}
            elif isinstance(item, list):
                return [convert(v) for v in item]
            return item
        
        serializable_data = convert(data)
        
        # Validate JSON serialization
        try:
            json.dumps(serializable_data)
        except TypeError as e:
            raise ValueError(f"Data is not JSON serializable: {e}")
        
        return serializable_data

    def css(self):
        # TODO: make this robust to notebook reloads
        # for now, just be ok with double-loading css
        # ip = get_ipython()
        # if ip is not None:
        #     if not getattr(ip, css_key, False):
        css = pathlib.Path(__file__).parent / "index.css"
        _css = pathlib.Path(css).read_text()
        # TODO: a more elegant way to do this?
        # Remove any existing topk-widget-css style tags
        display(HTML("<script>document.querySelectorAll('style#" + css_id + "').forEach(e => e.remove())</script>"))
        time.sleep(0.1)
        display(HTML("<style id='" + css_id + "'>" + _css + "</style>"))
        # setattr(ip, css_key, True)


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.css()


    js = pathlib.Path(__file__).parent / "index.js"
    _esm = pathlib.Path(js).read_text()
    
    

