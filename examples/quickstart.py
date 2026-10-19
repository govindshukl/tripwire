"""Judge one news passage about a (fictional) monitored company."""
import json
from tripwire import Tripwire, load_questions

tw = Tripwire(adapter="govindshukl/Tripwire-AdverseMedia-4B")
state = {
    "target": {"name": "Halvorsen Marine ASA", "type": "company", "country": "NO", "sector": "Shipping",
               "description": "Norwegian operator of offshore supply vessels.", "why_tracked": "Monitored company"},
    "article": {"headline": "Halvorsen Marine misses bond coupon, hires restructuring adviser",
                "publisher": "example-news.example", "publisher_tier": "unrated",
                "passage": "Halvorsen Marine ASA did not pay the NOK 48 million coupon due on its 2027 bonds on "
                           "Monday and has appointed an adviser to negotiate with bondholders, the company said."}}
answers = tw.decide(state, load_questions(mention="Halvorsen Marine"))
print(json.dumps(tw.adverse(answers), indent=1))
print(json.dumps({k: answers[k] for k in ("same_entity", "direction", "severity")}, indent=1))
