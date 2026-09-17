"""Generate synthetic Capital One-style KPI investigation data.

The data is fake, but the table shapes, metric definitions, segment fields,
complaint fields, and pipeline defects are designed to resemble real analyst
work in a credit-card risk/product analytics environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import string

import numpy as np
import pandas as pd

from config import DATA_DIR


SEED = 461
N_ACCOUNTS = 12_000
START_DATE = pd.Timestamp("2026-01-01")
END_DATE = pd.Timestamp("2026-08-31")
SPIKE_START = pd.Timestamp("2026-08-12")
OUTPUT_DIR = DATA_DIR


@dataclass(frozen=True)
class MerchantTemplate:
    category: str
    names: tuple[str, ...]
    avg_amount: float
    sigma: float


MERCHANTS = (
    MerchantTemplate(
        "grocery",
        ("FRESH MART", "MARKET BASKET", "CITY GROCERY", "HARVEST FOODS", "GREEN AISLE"),
        52,
        0.45,
    ),
    MerchantTemplate(
        "restaurant",
        ("TST* NORTH GRILL", "SQ *RIVER CAFE", "URBAN NOODLE", "LINDEN COFFEE", "BAY TACOS"),
        31,
        0.55,
    ),
    MerchantTemplate(
        "gas",
        ("FUEL STOP 0421", "METRO GAS", "HIGHWAY MART", "QUICKFUEL", "PARKWAY ENERGY"),
        44,
        0.35,
    ),
    MerchantTemplate(
        "online_retail",
        ("AMZN MKTP US", "SHOPMART ONLINE", "PAYPAL *CITYGEAR", "WEBSTORE 9A", "MARKETPLACE PAY"),
        68,
        0.65,
    ),
    MerchantTemplate(
        "subscription",
        ("STREAMLY MONTHLY", "MUSICBOX SUB", "CLOUDSPACE PRO", "FITAPP PREMIUM", "NEWSPLUS DIGITAL"),
        17,
        0.25,
    ),
    MerchantTemplate(
        "travel",
        ("AIRLINE TICKET", "HOTEL RESERVATION", "RIDESHARE TRIP", "TRAVELBOOKING", "INTL CAFE"),
        184,
        0.85,
    ),
    MerchantTemplate(
        "health",
        ("WELLCARE PHARMACY", "CITY CLINIC", "DENTAL PARTNERS", "VISION CENTER", "RX DIRECT"),
        74,
        0.55,
    ),
    MerchantTemplate(
        "education",
        ("CAMPUS BOOKSTORE", "ONLINE COURSE", "STUDENT SERVICES", "CERT EXAM FEE", "UNIV PARKING"),
        89,
        0.70,
    ),
)

STATE_BY_REGION = {
    "Northeast": ("NY", "NJ", "PA", "MA", "CT", "MD", "VA"),
    "South": ("TX", "FL", "GA", "NC", "TN", "LA", "SC"),
    "Midwest": ("MN", "IL", "OH", "MI", "WI", "MO", "IN"),
    "West": ("CA", "WA", "AZ", "CO", "OR", "NV", "UT"),
}

PRODUCTS = ("cash_rewards", "travel_rewards", "student_card", "secured_card", "venture_style")
CUSTOMER_SEGMENTS = ("student", "young_professional", "mass_market", "affluent")
CHANNELS = ("card_present", "web", "mobile", "recurring")


def rng_choice(rng: np.random.Generator, values, probs=None, size=None):
    return rng.choice(list(values), p=probs, size=size)


def random_code(rng: np.random.Generator, n: int = 4) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(rng.choice(list(alphabet), size=n))


def month_starts() -> list[pd.Timestamp]:
    return list(pd.date_range(START_DATE, END_DATE, freq="MS"))


def generate_accounts(rng: np.random.Generator) -> pd.DataFrame:
    account_ids = [f"ACCT{100000000 + i}" for i in range(N_ACCOUNTS)]
    product = rng_choice(
        rng,
        PRODUCTS,
        probs=(0.34, 0.23, 0.17, 0.11, 0.15),
        size=N_ACCOUNTS,
    )
    segment = rng_choice(
        rng,
        CUSTOMER_SEGMENTS,
        probs=(0.17, 0.29, 0.39, 0.15),
        size=N_ACCOUNTS,
    )
    fico_band = []
    credit_limits = []
    balances = []
    for prod, seg in zip(product, segment):
        low_fico_prob = {
            "student_card": 0.48,
            "secured_card": 0.78,
            "cash_rewards": 0.28,
            "travel_rewards": 0.15,
            "venture_style": 0.18,
        }[prod]
        if seg == "affluent":
            low_fico_prob *= 0.45
        elif seg == "student":
            low_fico_prob *= 1.25
        low_fico = rng.random() < min(low_fico_prob, 0.9)
        fico_band.append("<=660" if low_fico else ">660")

        if prod == "secured_card":
            limit = rng.normal(900, 260)
        elif prod == "student_card":
            limit = rng.normal(2100, 700)
        elif seg == "affluent":
            limit = rng.normal(14500, 4200)
        elif prod in {"travel_rewards", "venture_style"}:
            limit = rng.normal(9200, 3100)
        else:
            limit = rng.normal(5600, 2100)
        limit = float(np.clip(limit, 300, 30_000))
        credit_limits.append(round(limit, 2))

        util_mean = 0.51 if low_fico else 0.27
        if prod == "secured_card":
            util_mean += 0.08
        utilization = float(np.clip(rng.beta(2.2, 5.2) + util_mean / 4, 0.01, 0.97))
        balances.append(round(limit * utilization, 2))

    regions = rng_choice(rng, STATE_BY_REGION.keys(), probs=(0.24, 0.34, 0.22, 0.20), size=N_ACCOUNTS)
    states = [rng.choice(STATE_BY_REGION[r]) for r in regions]
    origin = rng_choice(rng, ("web", "mobile", "branch", "partner"), probs=(0.39, 0.37, 0.09, 0.15), size=N_ACCOUNTS)
    open_offsets = rng.integers(0, (START_DATE - pd.Timestamp("2018-01-01")).days, size=N_ACCOUNTS)
    open_dates = pd.Timestamp("2018-01-01") + pd.to_timedelta(open_offsets, unit="D")
    active = rng.choice([1, 0], p=[0.965, 0.035], size=N_ACCOUNTS)

    return pd.DataFrame(
        {
            "account_id": account_ids,
            "open_date": open_dates,
            "product_type": product,
            "customer_segment": segment,
            "fico_band": fico_band,
            "region": regions,
            "state": states,
            "channel_origin": origin,
            "active_flag": active,
            "current_balance": balances,
            "credit_limit": credit_limits,
        }
    )


def generate_monthly_snapshots(accounts: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    months = month_starts()
    for rec in accounts.itertuples(index=False):
        base_balance = rec.current_balance
        limit = rec.credit_limit
        dpd_base = 0.018 if rec.fico_band == ">660" else 0.064
        if rec.product_type == "secured_card":
            dpd_base += 0.015
        if rec.customer_segment == "student":
            dpd_base += 0.008

        for month in months:
            if pd.Timestamp(rec.open_date) > month + pd.offsets.MonthEnd(0):
                continue
            if rec.active_flag == 0 and month >= pd.Timestamp("2026-06-01") and rng.random() < 0.65:
                active = 0
            else:
                active = 1
            seasonal = 1.0 + 0.05 * math.sin(month.month / 12 * 2 * math.pi)
            balance_noise = rng.normal(1.0, 0.13)
            statement_balance = float(np.clip(base_balance * seasonal * balance_noise, 0, limit * 1.05))
            delinquency_prob = dpd_base
            if month >= pd.Timestamp("2026-08-01") and rec.fico_band == "<=660":
                delinquency_prob += 0.012
            is_30dpd = int(active and rng.random() < delinquency_prob)
            if is_30dpd:
                days_past_due = int(rng.choice([30, 45, 60, 75, 90, 120], p=[0.50, 0.20, 0.13, 0.08, 0.06, 0.03]))
            else:
                days_past_due = int(rng.choice([0, 0, 0, 5, 10, 15], p=[0.80, 0.08, 0.04, 0.04, 0.025, 0.015]))
            charge_off_balance = 0.0
            if days_past_due >= 120 and rng.random() < 0.35:
                charge_off_balance = round(statement_balance * rng.uniform(0.55, 1.0), 2)

            rows.append(
                {
                    "snapshot_month": month.strftime("%Y-%m"),
                    "account_id": rec.account_id,
                    "active_flag": active,
                    "statement_balance": round(statement_balance, 2),
                    "credit_limit": round(limit, 2),
                    "utilization_rate": round(statement_balance / limit if limit else 0, 4),
                    "days_past_due": days_past_due,
                    "is_30dpd": is_30dpd,
                    "charge_off_balance": charge_off_balance,
                    "source_system": "card_servicing_platform",
                    "snapshot_loaded_at": (month + pd.Timedelta(days=1, hours=3)).isoformat(),
                }
            )
    return pd.DataFrame(rows)


def merchant_descriptor(template: MerchantTemplate, rng: np.random.Generator) -> str:
    name = rng.choice(template.names)
    style = rng.choice(["plain", "terminal", "processor", "web"], p=[0.42, 0.22, 0.22, 0.14])
    if style == "terminal":
        return f"{name} {rng.integers(1000, 9999)}"
    if style == "processor":
        return f"SQ *{name[:15]} {rng.integers(100, 999)}"
    if style == "web":
        return f"{name}.COM*{random_code(rng, 5)}"
    return name


def transaction_amount(template: MerchantTemplate, rng: np.random.Generator) -> float:
    amount = rng.lognormal(mean=math.log(template.avg_amount), sigma=template.sigma)
    return round(float(np.clip(amount, 2.49, 2200)), 2)


def dispute_probability(rec, month: pd.Timestamp, category: str, channel: str, rng: np.random.Generator) -> float:
    base = 0.0105
    if rec.fico_band == "<=660":
        base += 0.0035
    if rec.customer_segment in {"student", "young_professional"}:
        base += 0.002
    if category == "travel":
        base += 0.006
    if channel == "mobile":
        base += 0.002
    if month >= SPIKE_START and category == "travel":
        base += 0.010
    if month >= SPIKE_START and category == "travel" and channel == "mobile":
        base += 0.025
    if month >= SPIKE_START and category == "travel" and channel == "mobile" and rec.fico_band == "<=660":
        base += 0.100
    if month >= SPIKE_START and category == "travel" and rec.customer_segment in {"student", "young_professional"}:
        base += 0.040
    return min(base, 0.18)


def fraud_probability(category: str, channel: str, rec) -> float:
    base = 0.004
    if channel in {"web", "mobile"}:
        base += 0.003
    if category in {"online_retail", "travel"}:
        base += 0.003
    if rec.fico_band == "<=660":
        base += 0.001
    return min(base, 0.06)


def generate_transactions(accounts: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    active_accounts = accounts[accounts["active_flag"] == 1].reset_index(drop=True)
    acct_lookup = {r.account_id: r for r in active_accounts.itertuples(index=False)}
    rows = []
    tx_counter = 1
    months = month_starts()
    merchant_probs = np.array([0.18, 0.19, 0.10, 0.20, 0.10, 0.10, 0.08, 0.05])
    merchant_probs = merchant_probs / merchant_probs.sum()

    for month in months:
        month_end = month + pd.offsets.MonthEnd(0)
        days_in_month = month_end.day
        for rec in active_accounts.itertuples(index=False):
            if pd.Timestamp(rec.open_date) > month_end:
                continue
            base_lambda = {
                "student": 7.0,
                "young_professional": 9.2,
                "mass_market": 8.0,
                "affluent": 10.6,
            }[rec.customer_segment]
            if rec.product_type in {"travel_rewards", "venture_style"}:
                base_lambda += 1.5
            n_purchases = int(rng.poisson(base_lambda))
            for _ in range(n_purchases):
                day = int(rng.integers(1, days_in_month + 1))
                tx_date = pd.Timestamp(year=month.year, month=month.month, day=day)
                template = rng.choice(MERCHANTS, p=merchant_probs)
                channel = rng_choice(rng, CHANNELS, probs=(0.43, 0.25, 0.22, 0.10))
                if template.category == "subscription":
                    channel = "recurring"
                if template.category == "travel" and month >= pd.Timestamp("2026-07-01"):
                    channel = rng_choice(rng, CHANNELS, probs=(0.25, 0.25, 0.40, 0.10))
                amount = transaction_amount(template, rng)
                posted_lag = int(rng.choice([0, 1, 2, 3, 4], p=[0.08, 0.45, 0.32, 0.12, 0.03]))
                posted_date = tx_date + pd.Timedelta(days=posted_lag)
                batch_date = posted_date if posted_date <= END_DATE else END_DATE
                is_disputed = int(rng.random() < dispute_probability(rec, tx_date, template.category, channel, rng))
                is_fraud = int(is_disputed and rng.random() < fraud_probability(template.category, channel, rec))

                if month >= SPIKE_START and template.category == "travel" and channel == "mobile":
                    source_system = "dispute_platform"
                else:
                    source_system = rng_choice(
                        rng,
                        ("card_processor", "card_processor", "card_processor", "payment_gateway"),
                    )
                batch_id = f"{source_system.upper()}_{batch_date.strftime('%Y%m%d')}_{rng.integers(1, 7):02d}"
                rows.append(
                    {
                        "transaction_id": f"TX{tx_counter:010d}",
                        "source_transaction_id": f"SRC{tx_counter:010d}",
                        "account_id": rec.account_id,
                        "transaction_date": tx_date.date().isoformat(),
                        "posted_date": posted_date.date().isoformat(),
                        "merchant_category": template.category,
                        "merchant_name": merchant_descriptor(template, rng),
                        "channel": channel,
                        "transaction_amount": amount,
                        "transaction_type": "purchase",
                        "is_disputed": is_disputed,
                        "is_fraud_claim": is_fraud,
                        "payment_failed": 0,
                        "ingestion_batch_id": batch_id,
                        "source_system": source_system,
                        "created_at": (batch_date + pd.Timedelta(hours=int(rng.integers(1, 6)), minutes=int(rng.integers(0, 60)))).isoformat(),
                    }
                )
                tx_counter += 1

            if rng.random() < 0.88:
                pay_day = int(rng.integers(12, min(days_in_month, 28) + 1))
                pay_date = pd.Timestamp(year=month.year, month=month.month, day=pay_day)
                fail_prob = 0.018 + (0.017 if rec.fico_band == "<=660" else 0)
                if month >= SPIKE_START and rec.channel_origin == "mobile":
                    fail_prob += 0.025
                payment_failed = int(rng.random() < fail_prob)
                amount = round(float(np.clip(rec.current_balance * rng.uniform(0.08, 0.55), 25, 4500)), 2)
                rows.append(
                    {
                        "transaction_id": f"TX{tx_counter:010d}",
                        "source_transaction_id": f"SRC{tx_counter:010d}",
                        "account_id": rec.account_id,
                        "transaction_date": pay_date.date().isoformat(),
                        "posted_date": (pay_date + pd.Timedelta(days=1)).date().isoformat(),
                        "merchant_category": "payment",
                        "merchant_name": "AUTOPAY THANK YOU" if rec.channel_origin == "mobile" else "ONLINE PAYMENT THANK YOU",
                        "channel": rec.channel_origin if rec.channel_origin in {"web", "mobile"} else "web",
                        "transaction_amount": -amount if not payment_failed else 0.0,
                        "transaction_type": "payment",
                        "is_disputed": 0,
                        "is_fraud_claim": 0,
                        "payment_failed": payment_failed,
                        "ingestion_batch_id": f"PAYMENT_GATEWAY_{pay_date.strftime('%Y%m%d')}_01",
                        "source_system": "payment_gateway",
                        "created_at": (pay_date + pd.Timedelta(days=1, hours=2)).isoformat(),
                    }
                )
                tx_counter += 1

            if rng.random() < (0.045 if rec.fico_band == "<=660" else 0.018):
                fee_date = month + pd.Timedelta(days=int(rng.integers(3, days_in_month)))
                fee_type = rng.choice(
                    ("LATE PAYMENT FEE", "RETURNED PAYMENT FEE", "CASH ADVANCE FEE", "FOREIGN TRANSACTION FEE"),
                    p=(0.42, 0.20, 0.10, 0.28),
                )
                if month >= SPIKE_START and rec.customer_segment in {"student", "young_professional"}:
                    fee_type = rng.choice(("FOREIGN TRANSACTION FEE", "LATE PAYMENT FEE"), p=(0.68, 0.32))
                rows.append(
                    {
                        "transaction_id": f"TX{tx_counter:010d}",
                        "source_transaction_id": f"SRC{tx_counter:010d}",
                        "account_id": rec.account_id,
                        "transaction_date": fee_date.date().isoformat(),
                        "posted_date": (fee_date + pd.Timedelta(days=1)).date().isoformat(),
                        "merchant_category": "fee",
                        "merchant_name": fee_type,
                        "channel": "system",
                        "transaction_amount": round(float(rng.choice([1.86, 2.75, 12.99, 25.0, 29.0, 35.0])), 2),
                        "transaction_type": "fee",
                        "is_disputed": int(rng.random() < 0.08),
                        "is_fraud_claim": 0,
                        "payment_failed": 0,
                        "ingestion_batch_id": f"CARD_PROCESSOR_{fee_date.strftime('%Y%m%d')}_FEE",
                        "source_system": "card_processor",
                        "created_at": (fee_date + pd.Timedelta(days=1, hours=4)).isoformat(),
                    }
                )
                tx_counter += 1

    tx = pd.DataFrame(rows)

    # Pipeline defect: replayed file creates duplicated source transactions for disputed
    # travel/mobile records after the August spike begins. IDs differ, source IDs match.
    defect_mask = (
        (pd.to_datetime(tx["transaction_date"]) >= SPIKE_START)
        & (tx["merchant_category"] == "travel")
        & (tx["channel"] == "mobile")
        & (tx["is_disputed"] == 1)
        & (tx["transaction_type"] == "purchase")
    )
    replay = tx[defect_mask].sample(frac=0.65, random_state=SEED).copy()
    if not replay.empty:
        replay_count = len(replay)
        replay["transaction_id"] = [f"TX{tx_counter + i:010d}" for i in range(replay_count)]
        replay["ingestion_batch_id"] = "DISPUTE_PLATFORM_20260818_REPLAY_01"
        replay["created_at"] = pd.Timestamp("2026-08-18T04:17:00").isoformat()
        replay["source_system"] = "dispute_platform"
        tx = pd.concat([tx, replay], ignore_index=True)

    # Data defect: missing categories from one card processor batch.
    miss_mask = (
        (pd.to_datetime(tx["posted_date"]).between("2026-07-24", "2026-07-26"))
        & (tx["source_system"] == "card_processor")
        & (tx["transaction_type"] == "purchase")
    )
    missing_idx = tx[miss_mask].sample(frac=0.18, random_state=SEED + 1).index
    tx.loc[missing_idx, "merchant_category"] = ""

    return tx.sort_values(["posted_date", "transaction_id"]).reset_index(drop=True)


# Complaint narrative vocabulary.
#
# The pools are split by business context rather than by issue text alone. The
# earlier version chose wording from issue/sub_issue only, so a grocery dispute
# could be described as "the same travel charge appears twice" -- fine in
# aggregate, but it made the dashboard's representative-evidence rows contradict
# their own merchant_category. Splitting the vocabulary keeps every narrative
# consistent with the row it was generated from.
#
# Rule for the generic pools: they must never contain travel vocabulary, and
# never claim a mobile app interaction. A self-check test enforces this.

VAGUE_NARRATIVES = (
    "The statement is difficult to understand and I need help reading the recent activity.",
    "I contacted support but still do not understand this account activity.",
    "The app and the statement do not show the same thing and I would like someone to explain it.",
)

DUPLICATE_TRAVEL_NARRATIVES = (
    "The same airline charge appears twice and I cannot tell whether one is still pending.",
    "I was billed twice for the same hotel reservation and I need one of the charges removed.",
    "A duplicate-looking travel booking charge posted after my trip and I want it reviewed.",
)

DUPLICATE_GENERIC_NARRATIVES = (
    "The same charge appears twice on my statement and I cannot tell whether it is pending or posted.",
    "I was billed twice by the same merchant and I need one of the charges removed.",
    "A duplicate-looking charge posted this month and I would like it reviewed.",
)

DESCRIPTOR_TRAVEL_NARRATIVES = (
    "The travel booking descriptor on my statement is unclear and I cannot tell which airline it was.",
    "There is a charge from an international merchant that I cannot identify after my trip.",
    "The hotel charge shows an unfamiliar name and I cannot tell which property billed me.",
)

DESCRIPTOR_GENERIC_NARRATIVES = (
    "There is a purchase on my statement that I do not recognize and the merchant name is unclear.",
    "The merchant descriptor is confusing and I cannot tell which store charged me.",
    "I cannot identify the business behind this charge from the name shown on my statement.",
)

STATUS_TRAVEL_NARRATIVES = (
    "I disputed a travel charge weeks ago and the status still has not been updated.",
    "I need help understanding why my dispute about the airline charge was closed with explanation.",
    "My dispute about the hotel booking is still open and nobody has told me what happens next.",
)

STATUS_GENERIC_NARRATIVES = (
    "I disputed this charge weeks ago and the status still has not been updated.",
    "I need help understanding why my dispute was closed with explanation.",
    "My dispute is still open and nobody has told me what happens next.",
)

MOBILE_DISPUTE_TRAVEL_NARRATIVES = (
    "I submitted a dispute for a duplicate travel charge in the mobile app but the status never changed.",
    "The app will not let me track the dispute I filed for my airline booking.",
    "I tried to file a dispute for a hotel charge on my phone and the app kept failing.",
)

MOBILE_DISPUTE_GENERIC_NARRATIVES = (
    "I submitted a dispute in the mobile app but the status never changed.",
    "The app will not let me track the dispute I filed and the statement shows something different.",
    "I tried to file a dispute on my phone and the app kept failing part way through.",
)

UNRECOGNIZED_TRAVEL_NARRATIVES = (
    "I do not recognize this charge from my trip and I would like it reviewed.",
    "There is a booking charge on my statement that I never authorized.",
)

UNRECOGNIZED_GENERIC_NARRATIVES = (
    "I do not recognize this purchase and I would like it reviewed.",
    "There is a charge on my statement that I never authorized.",
)

MOBILE_AUTOPAY_NARRATIVES = (
    "My autopay failed in the mobile app and now my account shows a late payment warning.",
    "The autopay looked successful on my phone but later showed as returned.",
    "The app said autopay was scheduled, but the balance did not update on time.",
)

ONLINE_PAYMENT_NARRATIVES = (
    "I paid online but the payment later showed as returned and now I have a late payment warning.",
    "The payment confirmation on the website did not match what my statement shows.",
    "I tried to make a payment online and it did not process on time.",
)

FEE_NARRATIVES_BY_TYPE = {
    "FOREIGN TRANSACTION FEE": (
        "I do not understand why a foreign transaction fee appeared after I used the card while traveling.",
        "An international purchase added a foreign transaction fee that I was not expecting.",
        "I was charged a foreign transaction fee and nobody explained what triggered it.",
    ),
    "LATE PAYMENT FEE": (
        "I was charged a late payment fee and I need a plain explanation of what triggered it.",
        "A late payment fee appeared even though I believed the balance was paid on time.",
    ),
    "RETURNED PAYMENT FEE": (
        "I was charged a returned payment fee even though I updated my bank account.",
        "A returned payment fee posted after my payment failed and I would like it reviewed.",
    ),
    "CASH ADVANCE FEE": (
        "A cash advance fee appeared on my statement and I do not understand what caused it.",
        "I was charged a cash advance fee for a transaction I thought was a normal purchase.",
    ),
}

GENERIC_FEE_NARRATIVES = (
    "The fee label on my statement is confusing and I cannot tell what triggered it.",
    "The statement shows a fee that I was not expecting and I need a plain explanation.",
)

TRAVEL_FRAUD_NARRATIVES = (
    "I received a fraud alert after a travel purchase and could not verify the transaction quickly.",
    "A purchase was declined while I was traveling and then a similar charge posted later.",
    "I do not recognize this merchant from my trip and worry the account was used without permission.",
)

FRAUD_NARRATIVES = (
    "I received a fraud alert and could not verify the transaction quickly.",
    "A purchase was declined and then a very similar charge posted later, which I cannot explain.",
    "I do not recognize this merchant and worry the account may have been used without permission.",
)

# Probability that a complaint is written vaguely instead of on-theme. Real
# complaint books contain plenty of unclear text; keeping some means the theme
# classifier is not scored against unrealistically tidy input. These stay
# generic, so they never contradict the row.
VAGUE_NARRATIVE_RATE = 0.06

# Probability of appending an urgency clause, carried over from the original.
URGENCY_RATE = 0.18


def _dispute_narrative_pool(is_travel: bool, is_mobile: bool, rng: np.random.Generator):
    """Pick the wording family for a disputed purchase.

    Mobile-channel disputes lean toward app-submission friction, which is the
    planted August story; every other intent still occurs so the corpus does not
    collapse onto one phrasing.
    """
    if is_mobile:
        intent = rng_choice(
            rng,
            ("mobile", "duplicate", "descriptor", "status", "unrecognized"),
            probs=(0.34, 0.20, 0.20, 0.16, 0.10),
        )
    else:
        intent = rng_choice(
            rng,
            ("duplicate", "descriptor", "status", "unrecognized"),
            probs=(0.28, 0.30, 0.22, 0.20),
        )

    pools = {
        "mobile": (MOBILE_DISPUTE_TRAVEL_NARRATIVES, MOBILE_DISPUTE_GENERIC_NARRATIVES),
        "duplicate": (DUPLICATE_TRAVEL_NARRATIVES, DUPLICATE_GENERIC_NARRATIVES),
        "descriptor": (DESCRIPTOR_TRAVEL_NARRATIVES, DESCRIPTOR_GENERIC_NARRATIVES),
        "status": (STATUS_TRAVEL_NARRATIVES, STATUS_GENERIC_NARRATIVES),
        "unrecognized": (UNRECOGNIZED_TRAVEL_NARRATIVES, UNRECOGNIZED_GENERIC_NARRATIVES),
    }
    travel_pool, generic_pool = pools[str(intent)]
    return travel_pool if is_travel else generic_pool


def complaint_narrative(
    issue: str,
    sub_issue: str,
    category: str,
    rng: np.random.Generator,
    *,
    channel: str | None = None,
    transaction_type: str | None = None,
    merchant_name: str = "",
    payment_failed: int = 0,
    is_disputed: int = 0,
    is_fraud_claim: int = 0,
) -> str:
    """Write a complaint narrative that agrees with the row it describes.

    The branch order mirrors the issue-assignment logic in
    :func:`generate_complaints`, so the narrative can never describe a different
    problem than the ``issue`` field claims. Within a branch, the wording is
    chosen from ``merchant_category``, ``channel`` and ``merchant_name`` so that
    an analyst reading a representative evidence row sees text consistent with
    the structured columns beside it.

    Only travel rows receive travel vocabulary, only mobile rows describe the
    app, and foreign-transaction-fee language is reserved for the fee type that
    actually carries that name.
    """
    merchant_name = str(merchant_name or "")
    is_travel = category == "travel"
    is_mobile = channel == "mobile"

    if rng.random() < VAGUE_NARRATIVE_RATE:
        pool = VAGUE_NARRATIVES
    elif payment_failed:
        pool = (
            MOBILE_AUTOPAY_NARRATIVES
            if (is_mobile or "AUTOPAY" in merchant_name)
            else ONLINE_PAYMENT_NARRATIVES
        )
    elif transaction_type == "fee" or "FEE" in merchant_name:
        # Keyed on the fee descriptor so foreign-transaction-fee wording only
        # appears on an actual foreign transaction fee.
        pool = FEE_NARRATIVES_BY_TYPE.get(merchant_name, GENERIC_FEE_NARRATIVES)
    elif is_fraud_claim:
        pool = TRAVEL_FRAUD_NARRATIVES if is_travel else FRAUD_NARRATIVES
    else:
        pool = _dispute_narrative_pool(is_travel, is_mobile, rng)

    sentence = str(rng_choice(rng, pool))
    if rng.random() < URGENCY_RATE:
        sentence += " I need this resolved before the next statement closes."
    return sentence


def generate_complaints(accounts: pd.DataFrame, transactions: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    purchase_tx = transactions[transactions["transaction_type"] == "purchase"].copy()
    disputed = purchase_tx[purchase_tx["is_disputed"] == 1]
    failed_payments = transactions[transactions["payment_failed"] == 1]
    fee_tx = transactions[transactions["transaction_type"] == "fee"]

    complaint_sources = []
    complaint_sources.append(disputed.sample(n=min(3100, len(disputed)), random_state=SEED))
    complaint_sources.append(failed_payments.sample(n=min(950, len(failed_payments)), random_state=SEED + 2))
    complaint_sources.append(fee_tx.sample(n=min(1250, len(fee_tx)), random_state=SEED + 3))
    seed_events = pd.concat(complaint_sources, ignore_index=True)

    acct = accounts.set_index("account_id")
    rows = []
    for i, rec in enumerate(seed_events.itertuples(index=False), 1):
        acc = acct.loc[rec.account_id]
        tx_date = pd.Timestamp(rec.transaction_date)
        received = tx_date + pd.Timedelta(days=int(rng.choice([1, 2, 3, 4, 5, 8, 12], p=[0.24, 0.25, 0.18, 0.13, 0.10, 0.06, 0.04])))
        if received > END_DATE:
            received = END_DATE

        if rec.payment_failed:
            issue = "Problem when making payments"
            sub_issue = rng.choice(("Problem with autopay", "Payment did not process", "Late fee after payment issue"))
        elif rec.transaction_type == "fee" or "FEE" in rec.merchant_name:
            issue = "Fees or interest"
            sub_issue = rng.choice(("Problem with fees", "Unexpected fee", "Foreign transaction fee"))
        elif rec.is_fraud_claim:
            issue = "Problem with fraud alerts or security"
            sub_issue = rng.choice(("Card was used without permission", "Problem with fraud alert", "Transaction declined"))
        else:
            issue = "Problem with a purchase shown on your statement"
            sub_issue = rng.choice(("Card was charged for something you did not purchase", "Credit card company isn't resolving a dispute", "Problem with merchant descriptor"))

        channel = rng.choice(("Web", "Phone", "Mobile app", "Referral"), p=(0.52, 0.25, 0.18, 0.05))
        timely = rng.choice(("Yes", "No"), p=(0.91, 0.09))
        response = rng.choice(
            ("Closed with explanation", "Closed with monetary relief", "Closed with non-monetary relief", "In progress"),
            p=(0.64, 0.16, 0.12, 0.08),
        )
        rows.append(
            {
                "complaint_id": f"CMP{2026000000 + i}",
                "account_id": rec.account_id,
                "related_transaction_id": rec.transaction_id,
                "date_received": received.date().isoformat(),
                "product": "Credit card",
                "sub_product": "General-purpose credit card or charge card",
                "issue": issue,
                "sub_issue": sub_issue,
                "complaint_narrative": complaint_narrative(
                    issue,
                    sub_issue,
                    rec.merchant_category,
                    rng,
                    channel=rec.channel,
                    transaction_type=rec.transaction_type,
                    merchant_name=rec.merchant_name,
                    payment_failed=int(rec.payment_failed),
                    is_disputed=int(rec.is_disputed),
                    is_fraud_claim=int(rec.is_fraud_claim),
                ),
                "submitted_via": channel,
                "company_response": response,
                "timely_response": timely,
                "merchant_category": rec.merchant_category,
                "channel": rec.channel,
                "customer_segment": acc["customer_segment"],
                "fico_band": acc["fico_band"],
                "state": acc["state"],
            }
        )
    return pd.DataFrame(rows).sort_values(["date_received", "complaint_id"]).reset_index(drop=True)


def metric_definitions() -> pd.DataFrame:
    rows = [
        {
            "metric_name": "dispute_rate",
            "business_definition": "Share of purchase transactions that generated a customer dispute.",
            "numerator": "count purchase transactions where is_disputed = 1",
            "denominator": "count purchase transactions",
            "grain": "monthly, with drilldowns by product, FICO band, segment, merchant category, channel, and region",
            "refresh_frequency": "daily batch with monthly KPI view",
            "owner": "card_disputes_analytics",
            "source_tables": "transactions, accounts",
            "known_limitations": "Inflated by duplicate source_transaction_id values if dispute-platform replay files are not deduplicated.",
        },
        {
            "metric_name": "fraud_claim_rate",
            "business_definition": "Share of purchase transactions associated with a fraud claim.",
            "numerator": "count purchase transactions where is_fraud_claim = 1",
            "denominator": "count purchase transactions",
            "grain": "monthly by product, merchant category, channel, and FICO band",
            "refresh_frequency": "daily batch with monthly KPI view",
            "owner": "fraud_strategy_analytics",
            "source_tables": "transactions, accounts",
            "known_limitations": "Fraud labels may lag customer dispute records.",
        },
        {
            "metric_name": "payment_failure_rate",
            "business_definition": "Share of payment attempts that failed or were returned.",
            "numerator": "count payment transactions where payment_failed = 1",
            "denominator": "count payment transactions",
            "grain": "monthly by channel, product, FICO band, and customer segment",
            "refresh_frequency": "daily batch with monthly KPI view",
            "owner": "payments_product_analytics",
            "source_tables": "transactions, accounts",
            "known_limitations": "Autopay status can lag the payment gateway event stream.",
        },
        {
            "metric_name": "delinquency_rate_30dpd_balance",
            "business_definition": "Balance-based 30+ day delinquency proxy for active credit-card accounts.",
            "numerator": "sum statement_balance where is_30dpd = 1",
            "denominator": "sum statement_balance for active accounts",
            "grain": "monthly by product, FICO band, customer segment, and region",
            "refresh_frequency": "monthly snapshot",
            "owner": "credit_risk_analytics",
            "source_tables": "account_monthly_snapshot, accounts",
            "known_limitations": "Synthetic proxy. Public disclosures use loans held for investment; this demo uses synthetic statement balance.",
        },
        {
            "metric_name": "net_charge_off_rate_proxy",
            "business_definition": "Annualized synthetic charge-off balance divided by average statement balance.",
            "numerator": "annualized sum charge_off_balance",
            "denominator": "average statement_balance",
            "grain": "monthly by product and FICO band",
            "refresh_frequency": "monthly snapshot",
            "owner": "credit_risk_analytics",
            "source_tables": "account_monthly_snapshot, accounts",
            "known_limitations": "Synthetic proxy for interview/demo use, not a regulatory reporting metric.",
        },
        {
            "metric_name": "complaint_rate",
            "business_definition": "Customer complaints per active account.",
            "numerator": "count complaints",
            "denominator": "count active accounts",
            "grain": "monthly by issue, product, FICO band, segment, channel, and region",
            "refresh_frequency": "daily complaint intake with monthly KPI view",
            "owner": "customer_experience_analytics",
            "source_tables": "complaints, accounts, account_monthly_snapshot",
            "known_limitations": "Complaint volume is influenced by channel adoption and reporting behavior.",
        },
        {
            "metric_name": "fee_complaint_share",
            "business_definition": "Share of complaints related to fees or interest.",
            "numerator": "count complaints where issue = Fees or interest",
            "denominator": "count complaints",
            "grain": "monthly by product, segment, FICO band, and channel",
            "refresh_frequency": "daily complaint intake with monthly KPI view",
            "owner": "customer_experience_analytics",
            "source_tables": "complaints",
            "known_limitations": "Issue labels depend on customer-selected complaint category.",
        },
    ]
    return pd.DataFrame(rows)


def write_ground_truth(accounts: pd.DataFrame, snapshots: pd.DataFrame, transactions: pd.DataFrame, complaints: pd.DataFrame) -> None:
    purchase = transactions[transactions["transaction_type"] == "purchase"].copy()
    purchase["month"] = pd.to_datetime(purchase["transaction_date"]).dt.to_period("M").astype(str)
    raw = purchase.groupby("month").agg(
        purchases=("transaction_id", "count"),
        disputed=("is_disputed", "sum"),
        unique_source_events=("source_transaction_id", "nunique"),
    )
    raw["raw_dispute_rate"] = raw["disputed"] / raw["purchases"]
    dedup = purchase.drop_duplicates("source_transaction_id")
    dedup_month = dedup.groupby("month").agg(
        purchases=("transaction_id", "count"),
        disputed=("is_disputed", "sum"),
    )
    dedup_month["deduped_dispute_rate"] = dedup_month["disputed"] / dedup_month["purchases"]
    replay_dupes = int(transactions["source_transaction_id"].duplicated().sum())
    missing_categories = int(
        (transactions["merchant_category"].isna() | (transactions["merchant_category"] == "")).sum()
    )
    aug_drivers = dedup[
        (dedup["month"] == "2026-08")
        & (dedup["is_disputed"] == 1)
    ].merge(accounts[["account_id", "fico_band", "customer_segment", "product_type", "region"]], on="account_id", how="left")
    driver_table = (
        aug_drivers.groupby(["merchant_category", "channel", "fico_band"])
        .size()
        .reset_index(name="disputed_purchase_count")
        .sort_values("disputed_purchase_count", ascending=False)
        .head(8)
    )
    top_complaint_terms = complaints[pd.to_datetime(complaints["date_received"]).dt.strftime("%Y-%m") == "2026-08"]["issue"].value_counts()

    # Narrative theme movement, measured with plain keyword probes.
    #
    # This is deliberately not the embedding classifier in
    # src/text_theme_analysis.py: ground truth should be checkable without
    # loading a model, and a probe that can be read in one line is harder to
    # argue with. The prose below is derived from these counts rather than
    # asserted, so this file cannot drift from the data it describes.
    narrative_probes = {
        "unclear merchant descriptor": r"descriptor|do not recognize|cannot identify|cannot tell which",
        "duplicate-looking charge": r"twice|duplicate",
        "travel-related charge": r"travel|airline|hotel|trip|international",
        "mobile app dispute friction": r"\bapp\b",
        "dispute status or resolution delay": r"dispute.*(?:status|closed with explanation|still open)",
        "foreign transaction fee": r"foreign transaction fee",
        "failed autopay": r"autopay",
        "fraud or security concern": r"fraud|without permission",
    }
    complaint_months = pd.to_datetime(complaints["date_received"]).dt.strftime("%Y-%m")
    july_text = complaints.loc[complaint_months == "2026-07", "complaint_narrative"].str.lower()
    august_text = complaints.loc[complaint_months == "2026-08", "complaint_narrative"].str.lower()
    probe_rows = []
    for label, pattern in narrative_probes.items():
        july_count = int(july_text.str.contains(pattern, regex=True).sum())
        august_count = int(august_text.str.contains(pattern, regex=True).sum())
        probe_rows.append(
            {
                "narrative_theme": label,
                "july_complaints": july_count,
                "august_complaints": august_count,
                "change": august_count - july_count,
            }
        )
    probe_table = pd.DataFrame(probe_rows).sort_values("change", ascending=False).reset_index(drop=True)
    rising = probe_table[probe_table["change"] > 0]["narrative_theme"].tolist()
    not_rising = probe_table[probe_table["change"] <= 0]["narrative_theme"].tolist()

    def _sentence(labels: list[str]) -> str:
        if not labels:
            return "none"
        if len(labels) == 1:
            return labels[0]
        return ", ".join(labels[:-1]) + f", and {labels[-1]}"

    def markdown_table(df: pd.DataFrame, index: bool = False, floatfmt: str = ".4f") -> str:
        if index:
            df = df.reset_index()
        out = df.copy()
        for col in out.columns:
            if pd.api.types.is_float_dtype(out[col]):
                out[col] = out[col].map(lambda x: format(x, floatfmt) if pd.notna(x) else "")
        cols = [str(c) for c in out.columns]
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for row in out.astype(str).itertuples(index=False):
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    monthly_rates = raw.join(dedup_month[["deduped_dispute_rate"]], how="left")
    issue_mix = top_complaint_terms.rename_axis("issue").reset_index(name="complaint_count")

    text = [
        "# Synthetic Ground Truth",
        "",
        "This file documents the intentional patterns planted in the synthetic data.",
        "",
        "## Intended Demo Story",
        "",
        "A credit-card dispute-rate alert fires in August 2026.",
        "",
        "The spike is partly caused by a replayed dispute-platform file, but a real business movement remains after deduplication.",
        "",
        "The remaining real movement is concentrated in travel merchant activity, mobile channel transactions, customers with FICO <=660, and student/young-professional segments.",
        "",
        f"Complaint narrative themes that rise from July to August: {_sentence(rising)}.",
        "",
        (
            f"Themes that do not rise from July to August: {_sentence(not_rising)}. "
            "They are present as background complaint volume, not as drivers of the "
            "August dispute-rate spike."
            if not_rising
            else "Every probed narrative theme rises from July to August, but the increases are "
            "far from equal: rank them by the change column below before calling anything a "
            "driver of the spike."
        ),
        "",
        "Narratives are generated from each row's merchant_category, channel, transaction_type, "
        "merchant_name and issue, so representative evidence rows agree with their structured columns: "
        "only travel rows use travel-specific purchase wording, mobile-app dispute friction is concentrated "
        "on mobile rows, and foreign-transaction-fee wording appears only on an actual FOREIGN TRANSACTION FEE row.",
        "",
        "## Planted Data Quality Issues",
        "",
        f"- Duplicate source transactions from replayed batch: {replay_dupes:,}",
        f"- Missing merchant categories from card-processor batch: {missing_categories:,}",
        "",
        "## Monthly Dispute Rate Check",
        "",
        markdown_table(monthly_rates, index=True),
        "",
        "## Top August Dispute Drivers After Deduplication",
        "",
        markdown_table(driver_table),
        "",
        "## August Complaint Issue Mix",
        "",
        markdown_table(issue_mix),
        "",
        "## Complaint Narrative Theme Movement (keyword probe, July vs August)",
        "",
        "Counts are complaints whose narrative matches a simple keyword probe. A narrative can "
        "match more than one probe, so these columns are not a partition and do not sum to the "
        "monthly complaint count.",
        "",
        "These probes are not the same measurement as the embedding theme classifier in "
        "src/text_theme_analysis.py, which assigns every complaint to exactly one theme. A probe "
        "can rise while the corresponding single-label theme falls: autopay wording rises here, "
        "but the classifier's failed_mobile_autopay theme also absorbs non-autopay payment "
        "failures, and payment complaints fall overall in August. Use the probes to confirm what "
        "the generator planted in the text, not as a substitute for the classifier output.",
        "",
        markdown_table(probe_table),
    ]
    (OUTPUT_DIR / "GROUND_TRUTH.md").write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    random.seed(SEED)
    rng = np.random.default_rng(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    accounts = generate_accounts(rng)
    snapshots = generate_monthly_snapshots(accounts, rng)
    transactions = generate_transactions(accounts, rng)
    complaints = generate_complaints(accounts, transactions, rng)
    definitions = metric_definitions()

    accounts.to_csv(OUTPUT_DIR / "accounts.csv", index=False)
    snapshots.to_csv(OUTPUT_DIR / "account_monthly_snapshot.csv", index=False)
    transactions.to_csv(OUTPUT_DIR / "transactions.csv", index=False)
    complaints.to_csv(OUTPUT_DIR / "complaints.csv", index=False)
    definitions.to_csv(OUTPUT_DIR / "metric_definitions.csv", index=False)
    write_ground_truth(accounts, snapshots, transactions, complaints)

    print("Synthetic data written to", OUTPUT_DIR)
    print("accounts:", f"{len(accounts):,}")
    print("account_monthly_snapshot:", f"{len(snapshots):,}")
    print("transactions:", f"{len(transactions):,}")
    print("complaints:", f"{len(complaints):,}")
    print("metric_definitions:", f"{len(definitions):,}")


if __name__ == "__main__":
    main()
