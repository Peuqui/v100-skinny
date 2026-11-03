# Welche Größe t(n) = (n mod m) - o erklärt, welche Prompts im Kontrollarm (Grenze 64) langsam sind?
# Modell: langsam genau dann, wenn 1 <= t <= 64 und t im selben Boot noch nicht vorkam (Warmup deckt 1..16 ab).
control = [  # (n, langsam) in Reihenfolge der Seeds 6200..6223
    (35738, 1), (35711, 1), (35675, 0), (35682, 0), (35745, 1), (35761, 1), (35670, 0), (35776, 0), (35806, 0),
    (35732, 1), (35633, 0), (35802, 0), (35796, 0), (35666, 0), (35856, 0), (35732, 0), (35881, 0), (35830, 0),
    (35873, 0), (35680, 0), (35776, 0), (35731, 1), (35990, 0), (35700, 0),
]
boot_woa3 = [(35758, 1), (35729, 1), (35665, 0), (35663, 0)]
candidate_cap16 = [  # Grenze 16: nur 6201 (35711) langsam
    (n, 1 if n == 35711 else 0) for n, _ in control
]
results = []
for modulus in (64, 128, 256, 512, 1024):
    for offset in range(0, modulus):
        for warm_max in (0, 16):
            errors = 0
            for run, cap in ((control, 64), (boot_woa3, 64)):
                seen = set(range(1, warm_max + 1))
                for n, slow in run:
                    t = (n % modulus) - offset
                    predicted = 0
                    if 1 <= t <= cap and t not in seen:
                        predicted = 1
                        seen.add(t)
                    errors += predicted != slow
            if errors == 0:
                results.append((modulus, offset, warm_max))
print("Passende (modulus, offset, warm_max) für Grenze 64:", results[:40], "Anzahl", len(results))
