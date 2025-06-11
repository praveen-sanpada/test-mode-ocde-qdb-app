def enrich_record(record: dict, source: str) -> str:
    lines = [f"{source.upper()} RECORD"]
    for k, v in record.items():
        if v in [None, '', []]:
            continue
        label = k.replace('_', ' ').capitalize()
        lines.append(f"{label}: {v}")
    return '\n'.join(lines)
