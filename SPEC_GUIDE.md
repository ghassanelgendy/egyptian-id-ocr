# Egyptian National ID Structure & API Specification 💳

This document provides a complete reference for the structure, decoding logic, and validation rules of the 14-digit Egyptian National ID card, matching the SignMe API specifications.

---

## 🔍 14-Digit Number Structure

The 14-digit National ID number contains encoded demographic registration data:

| Position(s) | Length | Example | Meaning |
| :--- | :--- | :--- | :--- |
| **1** | 1 digit | `2` | Century — `2` = born 1900–1999 · `3` = born 2000–2099 |
| **2–3** | 2 digits | `99` | Year of birth (YY) — combine with century for full year |
| **4–5** | 2 digits | `01` | Month of birth (MM) |
| **6–7** | 2 digits | `01` | Day of birth (DD) |
| **8–9** | 2 digits | `12` | Governorate registration code (see complete table below) |
| **10–13** | 4 digits | `3456` | Serial number — odd = **Male** · even = **Female** |
| **14** | 1 digit | `7` | Check digit — used to validate the full number |

### 📝 Worked Example — ID number `29901011234567`:
* **Digit 1** → century = 1900s
* **Digits 2–3 (99)** → year = 1999 (1900 + 99)
* **Digits 4–5 (01)** → month = January
* **Digits 6–7 (01)** → day = 1st (Birth Date = `1999-01-01`)
* **Digits 8–9 (12)** → registered in Dakahlia governorate
* **Digits 10–13 (3456)** → serial `3456` is even → **Female**
* **Digit 14** → check digit

---

## 🏛️ Governorate Registration Codes Reference

Digits 8–9 encode the governorate where the birth was registered (not necessarily the current residence). The gaps in the numbering scheme (e.g. 05–10, 20) are by design.

| Code | English Name | Arabic Name |
| :--- | :--- | :--- |
| **01** | Cairo | القاهرة |
| **02** | Alexandria | الإسكندرية |
| **03** | Port Said | بور سعيد |
| **04** | Suez | السويس |
| **11** | Damietta | دمياط |
| **12** | Dakahlia | الدقهلية |
| **13** | Sharqia | الشرقية |
| **14** | Qalyubia | القليوبية |
| **15** | Kafr El Sheikh | كفر الشيخ |
| **16** | Gharbia | الغربية |
| **17** | Menoufia | المنوفية |
| **18** | Beheira | البحيرة |
| **19** | Ismailia | الإسماعيلية |
| **21** | Giza | الجيزة |
| **22** | Beni Suef | بني سويف |
| **23** | Fayoum | الفيوم |
| **24** | Minya | المنيا |
| **25** | Assiut | أسيوط |
| **26** | Sohag | سوهاج |
| **27** | Qena | قنا |
| **28** | Aswan | أسوان |
| **29** | Luxor | الأقصر |
| **31** | Red Sea | البحر الأحمر |
| **32** | New Valley | الوادي الجديد |
| **33** | Matrouh | مطروح |
| **34** | North Sinai | شمال سيناء |
| **35** | South Sinai | جنوب سيناء |
| **88** | Foreign-born | خارج الجمهورية |

---

## 🧬 Gender Derivation

Gender is derived from the 4-digit serial (digits 10–13):
* **Odd** final digit (e.g. 1, 3, 5, 7, 9) $\rightarrow$ **Male** (`"Male"`)
* **Even** final digit (e.g. 0, 2, 4, 6, 8) $\rightarrow$ **Female** (`"Female"`)

*Note: The 14th check digit has no gender meaning and is purely mathematical.*

---

## 🚫 Common Field Handling Mistakes

1. **Storing `national_id` as an integer:** 
   Always store as `VARCHAR(14)` or `CHAR(14)`. Casting to a 32-bit integer causes instant overflow/truncation.
2. **Confusing registration vs. residential governorates:**
   The registered governorate (digits 8–9) never changes. The current residential governorate is printed on the address card layout. Always treat them separately.
3. **Not applying RTL direction for Arabic text:**
   Wrap all Arabic outputs with `<span lang="ar" dir="rtl">` in HTML to ensure correct word order rendering.
4. **Expecting `religion` and `marital_status` to always be present:**
   Religion was phased out of newer card generations starting ~2021. Always treat these fields as nullable.
5. **Hardcoding date comparisons:**
   Parse standard ISO `YYYY-MM-DD` strings locally with timezone-agnostic date objects (e.g., `DateOnly` in .NET or `T00:00:00` in JS) to prevent timezone offsets from shifting dates.
