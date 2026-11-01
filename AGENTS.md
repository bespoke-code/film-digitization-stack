# Guidelines

- The end goal (product) is described in [PRODUCT.md](PRODUCT.md). Respect the goal.
- Prefer standard libraries and pure implementations in rust.
- Do not go overboard with crates.
- The software must be supported on Linux and Mac (x86-64 and ARM architectures).
- Avoid comment bloat.
- User must be able to start and run the software on their machine.
- Performance is key. Optimize for speed.
- Use conventional commits. Full commit messages explain WHY something was done and 
  use up to 10 bullet points to explain what was done.
- No upstream CI/CD pipeline configurations. Locally, define pipelines in a file and 
  prefer using just to run parts or the full pipelines.
- ASD-STE100 for technical tests.
- DO NOT modify AGENTS.md or PRODUCT.md without prior approval.
- Create a README.md on the top level and always keep it short, up to date and start 
  with the WHY of the product, followed by an example input and output photos, and then 
  installation instructions.
