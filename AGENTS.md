# AI Core development standards

AI Core is an independent, reusable Python AI backend. EasyNet (Laravel) is
its initial client; keep client-specific business logic outside AI Core.

## Python conventions

- Follow PEP 8; use meaningful names and type hints.
- Prefer readable, explicit code and standard Python patterns. Do not copy
  Laravel architecture into Python.
- Use Pydantic for API request and response validation.
- Avoid unnecessary dependencies. Use the existing `.venv` interpreter.

## Architecture

- Apply separation of concerns and SOLID principles where useful. Keep HTTP
  endpoints thin; separate API handling, business logic, AI providers, and
  infrastructure as requirements grow.
- Use dependency injection where useful. Introduce abstractions and files only
  when real requirements justify them; these standards do not require an
  immediate folder restructure.
- Keep provider integration separable to support multiple LLM providers in the
  future, without implementing speculative providers or interfaces now.
- Avoid premature microservices and overengineering.

## Security

- Never hardcode credentials or secrets; use environment-based configuration.
- Validate all external inputs. Treat retrieved documents and LLM output as
  untrusted data; never rely on LLM instructions for authorization.
- Design data access and authorization for future tenant isolation; do not
  assume all clients may access the same data.
- Apply request timeouts and resource limits when adding external calls or
  resource-intensive operations.
- Never log secrets or sensitive document contents.
- Do not expose internal AI services publicly without appropriate
  authentication and network protection.
- Use synthetic or explicitly approved data only.

## Error handling

- Fail explicitly when external services are unavailable; do not silently
  ignore failures.
- Use clear application-level exceptions and translate them into safe API
  errors without exposing internal infrastructure details.
- Avoid broad exception handling that hides programming errors.

## Development workflow

- Inspect the existing implementation before editing and explain the intended
  change before implementation.
- Keep changes small and scoped to the requested task. Prefer one main
  implementation file per step, but use additional files when required for
  correctness, security, or justified separation of concerns.
- Preserve existing functionality; do not refactor unrelated code or create
  unnecessary files or abstractions.
- Do not install packages without explicit approval.
- Never modify system configuration or Docker services unless explicitly
  requested.
- Do not commit or push unless explicitly requested.

## Verification

- After Python changes, check syntax using the existing `.venv` interpreter.
  Documentation-only changes need documentation review, not a Python check.
- Use manual verification for the current POC. Do not create or run automated
  tests unless explicitly requested.
- Report changed files and verification results. Clearly distinguish verified
  behavior from assumptions and identify any checks that were not performed.

## Developer Ownership & Learning

For all future implementation tasks:

- Before editing, explain the intended changes, affected files, and data flow.
- Implement incrementally, preferably one main implementation file per step.
- After implementation, explain each changed file and the responsibility of
  each new method.
- Briefly relate unfamiliar Python concepts to PHP/Laravel equivalents where
  useful, while retaining standard Python design patterns.
- Explain important design decisions and why the chosen approach fits.
- Provide a short walkthrough of how data enters, moves through, and exits
  the changed code.
- Identify new dependencies, configuration requirements, security implications,
  and limitations; explicitly state when there are none.
- Provide manual verification steps and expected results, distinguishing
  those instructions from checks actually performed.
- Do not commit automatically. The developer reviews and accepts changes
  before committing; commit only when explicitly requested.
- Keep explanations concise. Focus on architecture, business logic, and
  important implementation details rather than trivial syntax.
