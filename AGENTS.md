## Code layout
Order the functions of a module, and the methods of a class, so that the file reads from top to bottom:

1. **Dunders first.** `__init__` and the other `__dunder__` methods open a class, `__init__` leading.
2. **Then public.** What a module or a class offers comes before how it does it: public functions and methods, then the private ones (`_`-prefixed).
3. **Callers before callees.** Within each group, a function comes before the functions it calls, so the reader meets a name where it is used and finds its definition further down. A helper shared by several callers goes after the first of them.


## Public API
A public function is complete when it is listed in the `__all__` of its module, exported from `jax_healpy/__init__.py`, listed in its page under `docs/api/`, tested and entered in the changelog.

Its signature and its conventions are those of healpy: departing from them is a decision (see Issues).


## Tests
Run the tests with `uv run pytest`: the tests marked `slow` are deselected by default. Build the documentation with `uv run --group docs sh docs/build.sh`. `CONTRIBUTING.md` has the details.

In the tests, patch with the `mocker` fixture of pytest-mock, never with `monkeypatch`. `mocker` (or `session_mocker`, `module_mocker`...) is the first argument of the test or of the fixture, before `request` and the parametrized arguments.

A fix starts with a test that reproduces the bug: run it and see it fail before writing the fix.

A test checks a result against a reference -- healpy, or an analytical value -- not against the output of the code it tests, and not only its shape or its finiteness. A function that is meant to be compiled, differentiated or batched is also tested under `jax.jit`, `jax.grad` or `jax.vmap`, and under both precisions, with the `x64` fixture, when the 32-bit and the 64-bit behaviors differ.

A test is never weakened to make it pass: loosening a tolerance, or adding a `skip`, an `xfail` or a `flaky` rerun, is a decision (see Issues).


## Performance
A performance claim comes with its measurement, in the pull request: the figures before and after the change, and how they were taken. A timing excludes the JIT compilation, is the minimum over several runs and comes from a process with a fixed number of threads. healpy and ducc0 are timed in a separate process, without JAX, whose thread pool competes with theirs.


## Documentation
The documentation describes the current state: what exists, how it works, and why it is built that way. The history belongs elsewhere: what was missing, the options weighed and the choice made go to the issue (see Issues); what a change replaces, or how something "used to" behave, to the pull request and its commit message. A limit is stated as it stands today, in the present tense, with the issue that will lift it when there is one.

The comments and the docstrings of the code follow the same rule: they say what the code does and why, not what it replaces or how it "used to" behave.

A change carries the documentation it makes untrue, in the same pull request, and every revision pushed to it keeps the two in step: never a follow-up. That is:

- `README.md` and `CONTRIBUTING.md`, where they describe what changed.
- the pages under `docs/`.
- `docs/changelog.md`: a change that a user of the package can observe gets an entry under `[Unreleased]`, in the section that fits (`Added`, `Changed`, `Fixed`...), ending with the PR number, as in `(#54)`. A change to the documentation, the tests or the CI gets none.


## Issues
The implementation choices are made in the issue: it states the problem, the options weighed and the one that is retained, with the reasons.

When a change needs a decision that is the human's to take, create an issue that explains the pros and cons of each option: the decision is taken there. Adding a dependency, changing a public signature, departing from healpy, breaking the compatibility and weakening a test are such decisions.

An attempt that was dropped without being the subject of a decision goes in a comment of the issue or of the pull request.


## Branches
Create a branch from `origin/main`, after a `git fetch`: the local `main` may be behind. Leave alone the branches, the worktrees and the untracked files that are not yours.

Name a branch `<type>/<description>`, in the style of conventionalbranch.org:
- The type is one of the Conventional Commits types listed under Committing: `feat/`, `fix/`, `docs/`, `perf/`, `refactor/`, `test/`, `chore/`, etc.
- The description is lowercase and hyphen-separated
- Examples: `feat/ducc0-backend`, `fix/alm2map-lmax`, `docs/migrating-from-healpy`


## Committing
The pre-commit hooks run on every commit and are the same checks the CI runs, so a commit that fails them would fail the CI too. Fix the findings rather than bypassing them with `--no-verify`.

Before committing, `git add` each file that is meant to be committed -- by path, never `git add -A`, which would sweep in unwanted files -- then run `uv run prek run`: it checks the staged files only, exactly as the commit hook will, and is far cheaper than a full sweep. A file git does not know yet is checked by nothing, so new files must be added first. The CI runs the hooks on the whole tree (`uv run prek run --all-files`).

Whenever you create a git commit, always format the message using Conventional Commits:

`<type>[(<scope>)]: <description>`

- type: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert
- scope: optional, lowercase, the affected module/package
- description: imperative mood, no period
- The whole subject line, type and scope included, stays under 72 chars
- The body should describe the change, not the motivation
- Add `BREAKING CHANGE:` in the footer for breaking changes

Never commit with a generic message like "update" or "fix stuff". The message should reflect the current implementation without referencing the previous versions or attempts and match the description of the change in the PR.


## Pull requests
A pull request carries one change: a defect met on the way, unrelated to it, goes in an issue of its own.

Always create pull requests using the GitHub CLI `gh`:
- Use `gh pr create --base <target-branch> --head <source-branch> --title <conventional-message> --body <description>`
- Follow conventional commit naming for PR titles: `<type>[(<scope>)]: <description>`
- Include a clear description explaining the purpose of the PR, the content of the change and a *why* section when the change isn't trivial
- Link the issue that the PR closes (`Closes #<number>`): the implementation choices are there and are not repeated
- Say what was run to check the change, and what was not

After each push, follow the CI with `gh pr checks <number> --watch` and fix what fails.

A pull request holds a single commit, whose subject is the PR title. Each revision amends it (`git commit --amend`) and is pushed with `git push --force-with-lease`: the message is updated to describe the whole final change, and the PR description is kept in step with it. The merge can then be done from the GitHub interface, which takes that message as it is. It is the human's to do: never merge a pull request, and never push to `main`.
