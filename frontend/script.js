document.addEventListener('DOMContentLoaded', () => {
    const checkBtn        = document.getElementById('check-btn');
    const resetBtn        = document.getElementById('reset-btn');
    const smsInput        = document.getElementById('sms-input');
    const resultContainer = document.getElementById('result-container');
    const btnText         = checkBtn.querySelector('.btn-text');
    const loader          = checkBtn.querySelector('.loader-dots');

    const resultBadge      = document.getElementById('result-badge');
    const resultTitle      = document.getElementById('result-title');
    const predictionLine   = document.getElementById('result-prediction-line');
    const confidenceFill   = document.getElementById('confidence-fill');
    const confidenceValue  = document.getElementById('confidence-value');
    const heuristicFill    = document.getElementById('heuristic-fill');
    const heuristicValue   = document.getElementById('heuristic-value');
    const highlightedText  = document.getElementById('highlighted-text');
    const reasonsList      = document.getElementById('reasons-list');
    const ruleFlagsBox     = document.getElementById('rule-flags');

    const sampleSelect     = document.getElementById('sample-select');
    const loadSampleBtn    = document.getElementById('load-sample-btn');
    const samplePreview    = document.getElementById('sample-preview');
    const samplesCount     = document.getElementById('samples-count');

    const API_URL = `${window.location.origin}/predict`;

    // -----------------------------------------------------------------------
    // Sample library — 28 messages across 5 categories
    // -----------------------------------------------------------------------

    const EXAMPLES = {

        // ── Classic Smishing ──────────────────────────────────────────────
        smish_amazon: {
            label: "Amazon prize + URL + urgency",
            text: "URGENT: You've WON a £500 Amazon gift card! Claim NOW before it EXPIRES: http://amaz0n-reward.tk/claim?id=9921 Reply STOP to opt out.",
        },
        smish_bank_kyc: {
            label: "Bank KYC suspension alert",
            text: "⚠ ALERT: Your HDFC Bank a/c ending 4821 is SUSPENDED due to KYC non-compliance. Verify NOW at http://hdfc-kyc-verify.net/update or your account will be permanently blocked within 24 hrs. Call 1800-266-4332.",
        },
        smish_fedex: {
            label: "FedEx redelivery fee + link",
            text: "FedEx: Your package #FX-948201 could not be delivered. A $2.99 redelivery fee is required. Pay now: http://fedex-redelivery.info/pay?id=948201 Link expires in 12 hours.",
        },
        smish_irs: {
            label: "IRS tax refund claim",
            text: "IRS NOTICE: A tax refund of $847.00 is pending in your name. To claim, verify your identity at http://irs-refund-portal.com/verify. Failure to claim within 48 hrs will result in forfeiture. Ref: TX-2024-0092.",
        },
        smish_paypal: {
            label: "PayPal unauthorized charge",
            text: "PayPal Security: An unauthorized charge of $249.99 was detected on your account from IP 185.xx.xx.xx. If you did not authorize this, click http://paypal-secure-alert.co/dispute immediately to reverse. Your account will be limited otherwise.",
        },
        smish_netflix: {
            label: "Netflix billing suspension",
            text: "NETFLIX: Your payment method was declined and your account will be suspended in 24 hours. Update your billing details at http://netflix-billing-update.cc/secure to keep your subscription. Act now to avoid interruption.",
        },

        // ── False Negatives — hard to catch ──────────────────────────────
        fn_gift_cards: {
            label: "Gift-card pretexting (no URL, no $)",
            text: "hey it's mike from the office — got a sec? need you to grab a couple of gift cards for the client meeting, ill pay you back at lunch",
        },
        fn_ceo_fraud: {
            label: "CEO wire transfer request",
            text: "Hi, it's David, I'm tied up in a board meeting. I need you to process an urgent wire transfer of $12,000 to a new vendor today — very sensitive, keep this between us and don't call, just reply here once it's done.",
        },
        fn_wrong_number: {
            label: "Wrong-number → crypto romance",
            text: "Hey sorry, wrong number lol — or maybe not? You seem interesting. I'm Sophia, I work in crypto trading. Made a lot this quarter. What do you do?",
        },
        fn_charity: {
            label: "Earthquake relief donation scam",
            text: "Hi, I'm volunteering for earthquake relief in Turkey. We urgently need donations to reach 500 families tonight. Any amount helps — please send what you can via Venmo @reliefnow2024. God bless you.",
        },
        fn_landlord: {
            label: "Landlord new bank details",
            text: "Hi, this is your landlord. My bank has changed so please transfer this month's rent to the new account: Sort code 20-14-53, Account no 43821901, Ref: your name. Let me know once it's done.",
        },
        fn_ssa_callback: {
            label: "SSA suspension → callback number",
            text: "This is an automated notice from the Social Security Administration. Your Social Security number has been suspended due to suspicious activity linked to a criminal case. To prevent an arrest warrant, call back immediately: 1-888-422-9041.",
        },

        // ── False Positives — ham that fires signals ──────────────────────
        fp_mum_transfer: {
            label: "Family transfer request (mum)",
            text: "Hi mum, urgent — can you transfer £20 to my account today? Bank app is down so I can't move it myself. Sort code same as before.",
        },
        fp_dinner_bill: {
            label: "Friend splitting dinner bill",
            text: "Hey! We need to settle last night's dinner — your share is $34. Can you send it to my account ASAP? I need to pay back the card by tonight. Venmo or bank transfer both fine.",
        },
        fp_real_otp: {
            label: "Real bank OTP message",
            text: "Your Bank of America verification code is 847291. This code expires in 10 minutes. Do not share this with anyone. If you did not request this, call us immediately at 1-800-432-1000.",
        },
        fp_doctor: {
            label: "Doctor appointment with portal link",
            text: "Reminder from Dr. Patel's clinic: Your appointment is tomorrow at 10:30 AM. Please confirm or reschedule at https://healthportal.drpatel.com/confirm/a8f2. Reply STOP to unsubscribe from reminders.",
        },
        fp_school_alert: {
            label: "School early-dismissal alert + link",
            text: "SCHOOL ALERT: Early dismissal today at 1:30 PM due to a severe weather warning in the area. Please arrange pickup or confirm your child will take the bus. Details at https://lakeviewschool.org/alerts. — Admin Office",
        },
        fp_boss_call: {
            label: "Boss asking for urgent call",
            text: "Hey — are you free for a quick call? Something urgent came up with the client and I need to loop you in before the 3 pm meeting. Call me when you can. Important.",
        },

        // ── Tricky / Borderline ───────────────────────────────────────────
        tricky_parcel: {
            label: "Parcel tracking via short-link",
            text: "Your parcel is out for delivery today between 2–6 PM. Track live: https://bit.ly/trk-88291. If you won't be home, reply with a safe place to leave it.",
        },
        tricky_spotify: {
            label: "Spotify subscription expiry notice",
            text: "Your Spotify Premium subscription expires in 3 days. To keep your music going without interruption, update your payment method at spotify.com/account. Reply STOP to cancel these reminders.",
        },
        tricky_recruiter: {
            label: "LinkedIn recruiter (Goldman Sachs)",
            text: "Hi, I came across your profile on LinkedIn. I'm a talent recruiter at Goldman Sachs and we have an Analyst position that matches your background. Please review the JD and apply at https://gs-careers-apply.net/ref/8821 — deadline this Friday.",
        },
        tricky_survey: {
            label: "Survey + Amazon voucher offer",
            text: "Congratulations! You've been selected for an exclusive 2-minute customer survey. Complete it in the next 24 hours and receive a $50 Amazon voucher as thanks. Start here: https://survey-rewards.co/s?id=7731",
        },
        tricky_friend_shortlink: {
            label: "Friend sharing playlist via bit.ly",
            text: "Hey! Here's that playlist I mentioned — https://bit.ly/spfy-vibes22 — it has all the tracks from the party. Let me know what you think, it goes hard 🎧",
        },
        tricky_bank_real: {
            label: "Legitimate bank fraud alert",
            text: "BARCLAYS FRAUD ALERT: A transaction of £349 at Steam Online was attempted on your card ending 7741 at 22:14. If this was NOT you, call 0345 734 5345 immediately. If it was you, no action needed.",
        },

        // ── Clearly Benign ────────────────────────────────────────────────
        ham_study: {
            label: "Study session invite",
            text: "Hey, are you coming to the study session tomorrow at 3 pm? Let me know if you need the lecture notes — I can share the folder.",
        },
        ham_flight: {
            label: "British Airways flight update",
            text: "British Airways: Your flight BA2491 to Madrid on 25 Apr is on time. Departs 14:35 from T5. Check in now at ba.com or via the BA app. Have a great flight!",
        },
        ham_library: {
            label: "Library book due reminder",
            text: "Reminder from Central Library: 'The Pragmatic Programmer' is due back on 28 Apr. Renew online at library.citycouncil.gov/renew or visit any branch. No fines if returned on time.",
        },
        ham_hike: {
            label: "Weekend hiking plans",
            text: "Are you around this weekend? Was thinking we could do that hike you mentioned — maybe Saturday morning? Weather looks decent, around 18°C. Let me know!",
        },
        ham_dentist: {
            label: "Dentist appointment confirmation",
            text: "Your appointment with Dr. Sarah Mills is confirmed for Thursday 24 Apr at 2:15 PM at City Dental Centre, Room 4. Reply YES to confirm or call 020-7123-4567 to reschedule. See you then!",
        },
    };

    // Show total count in the header
    const totalCount = Object.keys(EXAMPLES).length;
    if (samplesCount) samplesCount.textContent = `${totalCount} samples`;

    // -----------------------------------------------------------------------
    // Legacy data-example support (honesty card "Run this" buttons)
    // -----------------------------------------------------------------------
    const LEGACY_MAP = {
        urgent_smish:  'smish_amazon',
        hard_smish:    'fn_gift_cards',
        benign_chat:   'ham_study',
        false_positive:'fp_mum_transfer',
    };

    const FLAG_LABELS = {
        url_present:         "URL present",
        math_operators:      "math/operators",
        currency_symbol:     "currency symbol",
        phone_number:        "phone number",
        suspicious_keyword:  "suspicious keyword",
        long_message:        "long message",
        self_call_to_action: "reply / call CTA",
        leet_speak:          "leet-speak",
        embedded_email:      "embedded email",
    };

    // -----------------------------------------------------------------------
    // Sample selector wiring
    // -----------------------------------------------------------------------

    sampleSelect.addEventListener('change', () => {
        const key = sampleSelect.value;
        const ex  = EXAMPLES[key];
        if (!ex) return;

        smsInput.value = ex.text;
        loadSampleBtn.disabled = false;

        // Show preview label
        samplePreview.textContent = ex.label;
        samplePreview.style.display = 'block';
    });

    loadSampleBtn.addEventListener('click', () => {
        const key = sampleSelect.value;
        const ex  = EXAMPLES[key];
        if (!ex) return;
        smsInput.value = ex.text;
        analyze();
    });

    // -----------------------------------------------------------------------
    // Legacy "Run this" buttons in the honesty card (data-example attribute)
    // -----------------------------------------------------------------------
    document.querySelectorAll('[data-example]').forEach(btn => {
        btn.addEventListener('click', () => {
            const raw = btn.getAttribute('data-example');
            const key = LEGACY_MAP[raw] || raw;
            const ex  = EXAMPLES[key];
            if (ex) {
                smsInput.value = ex.text;
                analyze();
            }
        });
    });

    // -----------------------------------------------------------------------
    // Core button & keyboard wiring
    // -----------------------------------------------------------------------
    checkBtn.addEventListener('click', analyze);

    resetBtn.addEventListener('click', () => {
        smsInput.value = '';
        resultContainer.style.display = 'none';
        sampleSelect.value = '';
        loadSampleBtn.disabled = true;
        samplePreview.style.display = 'none';
        smsInput.focus();
    });

    smsInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
            e.preventDefault();
            analyze();
        }
    });

    // -----------------------------------------------------------------------
    // Core analysis flow
    // -----------------------------------------------------------------------
    async function analyze() {
        const text = smsInput.value.trim();
        if (!text) {
            alert('Please enter or select a message to analyze.');
            return;
        }
        setLoading(true);
        try {
            const response = await fetch(API_URL, {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify({ text }),
            });
            if (!response.ok) {
                const err = await response.json().catch(() => ({}));
                throw new Error(err.error || `API error ${response.status}`);
            }
            const data = await response.json();
            displayResult(data, text);
        } catch (error) {
            console.error('Error:', error);
            alert('Could not reach the backend. Make sure Flask is running on this port.\n\n' + error.message);
        } finally {
            setLoading(false);
        }
    }

    function setLoading(isLoading) {
        checkBtn.disabled = isLoading;
        btnText.style.display = isLoading ? 'none'  : 'block';
        loader.style.display  = isLoading ? 'flex'  : 'none';
    }

    // -----------------------------------------------------------------------
    // Result rendering
    // -----------------------------------------------------------------------
    function displayResult(data, originalText) {
        resultContainer.style.display = 'block';

        const isSmishing = data.is_smishing;
        const confPct = data.confidence != null
            ? Math.round(data.confidence * 100)
            : null;
        const heurPct = Math.round((data.heuristic_score || 0) * 100);

        if (isSmishing) {
            resultBadge.textContent = 'High risk';
            resultBadge.className = 'badge badge-danger';
            resultTitle.textContent = 'Likely smishing';
            resultTitle.style.color = '#f87171';
        } else {
            resultBadge.textContent = 'Likely safe';
            resultBadge.className = 'badge badge-legit';
            resultTitle.textContent = 'Looks legitimate';
            resultTitle.style.color = '#6ee7b7';
        }

        const reasons  = data.reasons || [];
        const top      = reasons.slice(0, 3).join(', ') || 'no rule signals matched';
        const confStr  = confPct != null ? confPct + '%' : 'n/a';
        predictionLine.textContent =
            `Prediction: ${data.prediction}   ·   Confidence: ${confStr}   ·   Top reasons: ${top}`;

        confidenceFill.style.background = isSmishing ? '#f87171' : '#6ee7b7';
        setTimeout(() => {
            confidenceFill.style.width = (confPct != null ? confPct : 0) + '%';
            confidenceValue.textContent = confStr;
            heuristicFill.style.width = heurPct + '%';
            heuristicValue.textContent = heurPct + '%';
        }, 60);

        highlightedText.innerHTML = renderHighlights(originalText, data.highlights || []);

        reasonsList.innerHTML = '';
        if (reasons.length === 0) {
            const li = document.createElement('li');
            li.className = 'empty';
            li.textContent = 'No rule-based signals fired on this message.';
            reasonsList.appendChild(li);
        } else {
            reasons.forEach(r => {
                const li = document.createElement('li');
                li.textContent = r;
                reasonsList.appendChild(li);
            });
        }

        ruleFlagsBox.innerHTML = '';
        const flags = data.rule_flags || {};
        Object.entries(FLAG_LABELS).forEach(([key, label]) => {
            const on  = !!flags[key];
            const div = document.createElement('div');
            div.className = 'flag-item' + (on ? ' on' : '');
            div.innerHTML = `<span class="flag-dot"></span><span>${label}</span>`;
            ruleFlagsBox.appendChild(div);
        });

        resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // -----------------------------------------------------------------------
    // Highlight renderer (XSS-safe)
    // -----------------------------------------------------------------------
    function renderHighlights(text, spans) {
        if (!spans || spans.length === 0) return escapeHtml(text);

        const sorted = [...spans].sort((a, b) => a.start - b.start);
        let out = '';
        let cursor = 0;
        for (const s of sorted) {
            if (s.start < cursor) continue;
            if (s.start > cursor) out += escapeHtml(text.slice(cursor, s.start));
            const segment = text.slice(s.start, s.end);
            out += `<mark class="${s.type}" title="${s.type.replace('_', ' ')}">${escapeHtml(segment)}</mark>`;
            cursor = s.end;
        }
        if (cursor < text.length) out += escapeHtml(text.slice(cursor));
        return out;
    }

    function escapeHtml(s) {
        return s
            .replace(/&/g,  '&amp;')
            .replace(/</g,  '&lt;')
            .replace(/>/g,  '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }
});
