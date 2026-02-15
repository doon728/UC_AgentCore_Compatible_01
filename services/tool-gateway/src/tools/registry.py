from __future__ import annotations

from typing import Callable, Dict, Type

from pydantic import BaseModel, ConfigDict



class ToolSpec(BaseModel):
    """
    Schema-driven tool definition.
    """
    name: str
    description: str
    input_model: Type[BaseModel]
    output_model: Type[BaseModel]
    handler: Callable[[BaseModel], BaseModel]
    model_config = ConfigDict(arbitrary_types_allowed=True)


# -----------------------
# Tool schemas (v1)
# -----------------------
class SearchKBInput(BaseModel):
    query: str


class SearchKBResult(BaseModel):
    id: str
    title: str
    score: float
    snippet: str


class SearchKBOutput(BaseModel):
    results: list[SearchKBResult]


class GetMemberInput(BaseModel):
    member_id: str


class Member(BaseModel):
    member_id: str
    first_name: str
    last_name: str
    dob: str
    plan: str


class GetMemberOutput(BaseModel):
    member: Member | None


class WriteCaseNoteInput(BaseModel):
    case_id: str
    note: str


class WriteCaseNoteOutput(BaseModel):
    written: bool
    note_id: str | None


# -----------------------
# Tool handlers (v1 stubs)
# -----------------------
def search_kb_handler(inp: SearchKBInput) -> SearchKBOutput:
    q = (inp.query or "").strip()
    if not q:
        return SearchKBOutput(results=[])

    return SearchKBOutput(
        results=[
            SearchKBResult(
                id="doc-001",
                title="Sample KB Doc",
                score=0.87,
                snippet=f"Matched on: {q}",
            )
        ]
    )


def get_member_handler(inp: GetMemberInput) -> GetMemberOutput:
    member_id = (inp.member_id or "").strip()
    if not member_id:
        return GetMemberOutput(member=None)

    return GetMemberOutput(
        member=Member(
            member_id=member_id,
            first_name="Jane",
            last_name="Doe",
            dob="1990-01-01",
            plan="SamplePlan",
        )
    )


def write_case_note_handler(inp: WriteCaseNoteInput) -> WriteCaseNoteOutput:
    case_id = (inp.case_id or "").strip()
    note = (inp.note or "").strip()
    if not case_id or not note:
        return WriteCaseNoteOutput(written=False, note_id=None)

    return WriteCaseNoteOutput(written=True, note_id="note-001")


# -----------------------
# Registry
# -----------------------
TOOL_REGISTRY: Dict[str, ToolSpec] = {
    "search_kb": ToolSpec(
        name="search_kb",
        description="Search the knowledge base for relevant documents.",
        input_model=SearchKBInput,
        output_model=SearchKBOutput,
        handler=search_kb_handler,
    ),
    "get_member": ToolSpec(
        name="get_member",
        description="Fetch a member record by member_id.",
        input_model=GetMemberInput,
        output_model=GetMemberOutput,
        handler=get_member_handler,
    ),
    "write_case_note": ToolSpec(
        name="write_case_note",
        description="Write a note to a case record.",
        input_model=WriteCaseNoteInput,
        output_model=WriteCaseNoteOutput,
        handler=write_case_note_handler,
    ),
}
