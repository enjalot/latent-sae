# import anywidget
import traitlets
import json
import ipyreact
import pathlib
from IPython.display import HTML, display

ipyreact.define_import_map({
  "d3-scale-chromatic": "https://esm.sh/d3-scale-chromatic@3.1.0"
})

class TopkVisWidget(ipyreact.Widget):
    # aaaaaaaaaaaaaaaaaa

    css = pathlib.Path(__file__).parent / "topk_vis.css"
    _css = pathlib.Path(css).read_text()
    display(HTML("<style>" + _css + "</style>"))

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


    js = pathlib.Path(__file__).parent / "topk_vis.js"
    _esm = pathlib.Path(js).read_text()
    

