"""design command implementation."""

from pathlib import Path

from .dramatize import _merge_character_into_cast, load_cast, save_cast
from .llm import Character


def run_design(
    workdir: Path,
    name: str,
    text: str | None,
    description: str | None,
    verbose: bool = False,
) -> None:
    """add a new character to the cast or update existing one."""
    cast = load_cast(workdir)
    cast_map = {c.name.lower(): c for c in cast}
    alias_map = {
        a.lower(): c.name.lower() for c in cast if c.aliases for a in c.aliases
    }

    # default values if not provided (though args are required in CLI)
    new_char = Character(
        name=name,
        description=description or "neutral voice",
        audition_line=text or "Hello, I am a new character.",
        aliases=[],
    )

    result = _merge_character_into_cast(new_char, cast_map, alias_map, verbose=True)

    final_cast = list(cast_map.values())

    # ensure Narrator is first if present
    narrator = next((c for c in final_cast if c.name.lower() == "narrator"), None)
    if narrator:
        final_cast.remove(narrator)
        final_cast.insert(0, narrator)

    save_cast(workdir, final_cast)

    print(f"design: character '{name}' {result}")


def cmd_design(args):
    run_design(
        Path(args.workdir),
        args.name,
        args.text,
        args.description,
        verbose=args.verbose,
    )
