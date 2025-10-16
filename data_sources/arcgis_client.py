# data_sources/arcgis_client.py
import requests

ARCGIS_PORTAL = "https://who.maps.arcgis.com/sharing/rest"

def _get_json(url, params=None, timeout=30):
    r = requests.get(url, params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()

def resolve_feature_service(item_id: str, layer_index: int = 0) -> str:
    """
    Return a FeatureServer layer URL (default layer 0) from an ArcGIS item id.
    Works when the item is a Service; if it’s a file, raises a clear error.
    """
    # 1) Item metadata — service items expose a direct "url"
    meta = _get_json(f"{ARCGIS_PORTAL}/content/items/{item_id}", {"f": "json"})
    service_url = meta.get("url", "")
    if service_url and ("FeatureServer" in service_url or "MapServer" in service_url):
        return f"{service_url.rstrip('/')}/{layer_index}"

    # 2) Item data — sometimes webmaps/apps list operationalLayers with URLs
    try:
        data = _get_json(f"{ARCGIS_PORTAL}/content/items/{item_id}/data", {"f": "json"})
        if isinstance(data, dict):
            u = (data.get("url") or "").strip()
            if "FeatureServer" in u or "MapServer" in u:
                return f"{u.rstrip('/')}/{layer_index}"
            for key in ("operationalLayers", "layers"):
                if isinstance(data.get(key), list):
                    for lyr in data[key]:
                        u = (lyr.get("url") or "").strip()
                        if "FeatureServer" in u or "MapServer" in u:
                            return f"{u.rstrip('/')}/{layer_index}"
    except Exception:
        pass  # /data might not be JSON; ignore

    raise RuntimeError(
        "Could not auto-resolve a FeatureServer URL from this item. "
        "This WHO item is likely a file download, not a live service. "
        "Please supply a REST FeatureServer URL manually."
    )

def query_layer(layer_url: str, params: dict) -> dict:
    """
    Call the ArcGIS FeatureServer/MapServer layer /query endpoint with params.
    Returns JSON/GeoJSON dict.
    """
    if not layer_url or "http" not in layer_url:
        raise ValueError(f"Invalid layer_url: {layer_url!r}")
    resp = requests.get(f"{layer_url.rstrip('/')}/query", params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()
