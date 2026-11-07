

def build_loni_script(loni_details):
    with open("utils/job_scripts/job_template.txt", 'r') as f:
        text = "".join(f.readlines())
        text = text.format(**loni_details)
    return text