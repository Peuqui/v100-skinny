Antwort auf sohom-cs in vllm-project/vllm#54260 (Entwurf, NICHT gesendet) — nach dem Öffnen des PRs, <PR> einsetzen

Thanks for checking this against current main. I have opened the fix as #<PR>, with a unit test that drives _update_states on a non-last rank and asserts the trim (it fails on unmodified main with [111, -1, -1, -1] vs [111]). A review would be very welcome.
