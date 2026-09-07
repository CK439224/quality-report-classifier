"""
Generate a small synthetic dataset of quality/failure report narratives.

This exists ONLY so the rest of the pipeline can be run end-to-end with zero
setup and no network access. It is template-based and will look repetitive
under close inspection -- see the top-level README's "Honest limitations"
section. Swap it for real data (src/fetch_data.py, or your own export) before
drawing any real conclusions from model performance.

Usage:
    python data/generate_sample_data.py --n-per-category 60 --seed 42 --out data/sample_reports.csv
"""
import argparse
import csv
import random

# Each category maps to a bank of sentence fragments that get recombined.
# Fragments are written in varied "voices" (terse, detailed, casual) on
# purpose, since real complaint/NCR narratives are written by many different
# people under different amounts of time pressure.
CATEGORY_FRAGMENTS = {
    "ENGINE": {
        "symptom": [
            "the engine stalled without warning while driving at highway speed",
            "engine makes a loud knocking noise on cold start",
            "check engine light came on and the vehicle started running rough",
            "engine lost power suddenly and would not accelerate",
            "white smoke coming from under the hood after about 10 minutes of driving",
            "engine would not turn over despite a fully charged battery",
            "noticed a strong burning smell coming from the engine bay",
            "engine idles very roughly and shakes the whole vehicle",
        ],
        "context": [
            "this happened at around {miles} miles",
            "issue started about {months} months after purchase",
            "dealer has seen the vehicle twice for this same issue",
            "problem occurs consistently in cold weather",
            "problem is intermittent and hard to reproduce on demand",
            "no warning lights appeared before the failure",
        ],
        "impact": [
            "had to pull over immediately for safety",
            "vehicle was towed to the dealership",
            "repair was covered under warranty",
            "still waiting on a diagnosis from the service department",
            "this is the second engine-related repair this year",
        ],
    },
    "ELECTRICAL SYSTEM": {
        "symptom": [
            "dashboard warning lights flicker on and off randomly",
            "power windows stopped working on both driver and passenger side",
            "vehicle would not start due to what the dealer called a wiring fault",
            "infotainment screen goes black intermittently while driving",
            "headlights dim and brighten on their own at low speed",
            "battery drains completely overnight even when the vehicle is off",
            "interior lights stay on continuously and cannot be turned off",
        ],
        "context": [
            "started happening around {miles} miles",
            "seems worse in humid weather",
            "technician found corrosion near a connector",
            "problem has occurred at least {count} times so far",
            "no aftermarket electronics have been installed",
        ],
        "impact": [
            "left the driver stranded in a parking lot",
            "required replacing the battery twice",
            "dealer replaced a wiring harness under warranty",
            "still unresolved after multiple visits",
        ],
    },
    "BRAKES": {
        "symptom": [
            "brake pedal felt spongy and went almost to the floor",
            "grinding noise when braking at low speed",
            "abs warning light stays on continuously",
            "vehicle pulled hard to one side when braking",
            "brakes felt like they were not engaging at all for a moment",
            "squealing noise from the front brakes even after new pads were installed",
        ],
        "context": [
            "happened during normal city driving at around {miles} miles",
            "occurred during a panic stop situation",
            "inspection found uneven wear on the rotors",
            "this is the second brake-related complaint for this vehicle",
        ],
        "impact": [
            "nearly resulted in a collision",
            "vehicle was taken in for emergency service",
            "dealer replaced rotors and pads under warranty",
            "reported to the dealership immediately after it happened",
        ],
    },
    "STEERING": {
        "symptom": [
            "steering wheel vibrates noticeably at highway speed",
            "steering felt loose and had excessive play",
            "power steering assist cut out briefly while turning",
            "clunking noise from the steering column when turning at low speed",
            "vehicle wandered on the highway and required constant correction",
        ],
        "context": [
            "noticed this around {miles} miles",
            "happens mostly on rough road surfaces",
            "dealer could not immediately duplicate the issue",
            "steering fluid level was found to be low on inspection",
        ],
        "impact": [
            "made the vehicle difficult to control",
            "was scheduled for a steering system inspection",
            "resulted in an alignment and steering rack replacement",
        ],
    },
    "SUSPENSION": {
        "symptom": [
            "loud clunking noise from the front suspension over bumps",
            "vehicle bounces excessively after hitting a pothole",
            "noticed uneven ride height on one side of the vehicle",
            "clicking noise from the rear suspension at low speed",
            "vehicle feels unstable and sways during turns",
        ],
        "context": [
            "started around {miles} miles",
            "worse on uneven or gravel roads",
            "inspection found a worn suspension component",
            "no prior accident or impact reported",
        ],
        "impact": [
            "component was replaced under warranty",
            "vehicle was deemed unsafe to drive until repaired",
            "still being evaluated by the service department",
        ],
    },
    "FUEL SYSTEM": {
        "symptom": [
            "strong fuel smell noticed near the rear of the vehicle",
            "engine hesitates and stumbles under acceleration",
            "fuel gauge reading is inconsistent with actual fuel level",
            "visible fuel leak near the fuel line was discovered",
            "vehicle would not accept a full tank without shutting off early",
        ],
        "context": [
            "noticed at around {miles} miles",
            "occurs mostly after refueling",
            "technician found a loose fuel line connection",
            "smell was strong enough to be noticeable from outside the vehicle",
        ],
        "impact": [
            "vehicle was not driven until inspected due to fire risk concerns",
            "fuel line was replaced under warranty",
            "reported immediately due to safety concerns",
        ],
    },
    "AIRBAGS": {
        "symptom": [
            "airbag warning light stays illuminated on the dashboard",
            "airbag deployed unexpectedly with no collision",
            "passenger airbag indicator shows off even with a passenger seated",
            "airbag did not deploy during a low speed collision",
        ],
        "context": [
            "warning light has been on since around {miles} miles",
            "dealer says a sensor may need replacement",
            "no diagnostic trouble code could be retrieved initially",
        ],
        "impact": [
            "vehicle safety is a serious concern given non-deployment",
            "scheduled for immediate inspection",
            "recall campaign was mentioned as a possibility by the dealer",
        ],
    },
    "TRANSMISSION": {
        "symptom": [
            "transmission slips noticeably between second and third gear",
            "hard, jarring shift when accelerating from a stop",
            "vehicle hesitates for a few seconds before engaging a gear",
            "grinding noise when shifting into reverse",
            "transmission fluid was found leaking underneath the vehicle",
        ],
        "context": [
            "began around {miles} miles",
            "worse when the transmission fluid is cold",
            "dealer has adjusted the transmission software once already",
            "problem has occurred over {count} separate occasions",
        ],
        "impact": [
            "vehicle was undriveable until repaired",
            "transmission was rebuilt under warranty",
            "still under evaluation by the service department",
        ],
    },
    "STRUCTURE": {
        "symptom": [
            "noticed visible rust forming along the frame rail",
            "door does not close flush with the body panel",
            "trunk lid does not align properly with the rear panel",
            "creaking and popping noises from the body over bumps",
            "paint is peeling near a body seam despite normal care",
        ],
        "context": [
            "first noticed around {miles} miles",
            "vehicle has never been in an accident",
            "dealer attributed it to a manufacturing defect during inspection",
        ],
        "impact": [
            "panel was replaced under warranty",
            "still awaiting a decision from the manufacturer",
            "raised concerns about long term structural integrity",
        ],
    },
    "TIRES": {
        "symptom": [
            "tire showed unusual wear pattern well before expected tread life",
            "tire lost pressure repeatedly despite no visible puncture",
            "vibration at highway speed traced to an out-of-round tire",
            "sidewall bulge appeared on one tire with normal use",
        ],
        "context": [
            "noticed at around {miles} miles",
            "all four tires were rotated and balanced with no improvement",
            "tire pressure monitoring system flagged the issue first",
        ],
        "impact": [
            "tire was replaced under manufacturer warranty",
            "raised safety concerns about a blowout at highway speed",
            "still waiting on a warranty determination",
        ],
    },
}

CONNECTORS = [
    "In addition, ",
    "Also, ",
    "Separately, ",
    "",
    "",
]


def build_narrative(rng: random.Random, fragments: dict) -> str:
    miles = rng.choice([3200, 8700, 15400, 22000, 31500, 44800, 58200, 71000])
    months = rng.choice([1, 2, 3, 4, 6, 9, 12])
    count = rng.choice([2, 3, 4, "several"])

    symptom = rng.choice(fragments["symptom"])
    context = rng.choice(fragments["context"]).format(miles=miles, months=months, count=count)
    impact = rng.choice(fragments["impact"])

    order = [symptom, context, impact]
    rng.shuffle(order[1:])  # keep symptom first, vary the rest

    sentence = f"{order[0].capitalize()}. {CONNECTORS[rng.randrange(len(CONNECTORS))]}{order[1]}. {order[2]}."
    return sentence.replace("..", ".").strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-category", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/sample_reports.csv")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = []
    report_id = 1
    for category, fragments in CATEGORY_FRAGMENTS.items():
        for _ in range(args.n_per_category):
            narrative = build_narrative(rng, fragments)
            rows.append({"report_id": report_id, "category": category, "narrative": narrative})
            report_id += 1

    rng.shuffle(rows)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["report_id", "category", "narrative"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} synthetic reports across {len(CATEGORY_FRAGMENTS)} categories to {args.out}")


if __name__ == "__main__":
    main()
