from .tag_specs import TAG_SPECS

def get_family_for_tag(tag: str) -> str | None:
    spec = TAG_SPECS.get(tag)
    if not spec:
        return None
    return spec.get("family")