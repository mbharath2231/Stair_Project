# STAIR device handoffs

When the user asks to hand off work to another device, include a daily
summary in the handoff by default. Create or update a Word study guide
for the day under `docs/handoffs/YYYY-MM-DD/` and list it in
[`docs/handoffs/README.md`](docs/handoffs/README.md).

Cover the project discussions for that day, completed work, verified
results and their limits, relevant code and outputs, and the next tasks
in order. Use repository-relative links so the guide can travel with
the checkout.

Include the guide in the commits pushed for the handoff. Keep large
generated outputs and observation caches in the repository's draft
release workflow. Record the release and extraction paths in
[`outputs/README.md`](outputs/README.md) and verify the pushed branch
and any new release asset checksums.
