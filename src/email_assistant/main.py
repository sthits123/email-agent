from typing import Literal

from langchain.chat_models import init_chat_model

from langgraph.graph import StateGraph, START, END
from langgraph.store.base import BaseStore
from langgraph.types import interrupt, Command

from email_assistant.tools import get_tools, get_tools_by_name
from email_assistant.tools.gmail.prompt_templates import GMAIL_TOOLS_PROMPT
from email_assistant.tools.gmail.gmail_tools import mark_as_read
from email_assistant.prompts import (
    triage_system_prompt,
    triage_user_prompt,
    agent_system_prompt_hitl_memory,
    default_triage_instructions,
    default_background,
    default_response_preferences,
    default_cal_preferences,
    MEMORY_UPDATE_INSTRUCTIONS,
    MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT,
)
from email_assistant.schemas import State, RouterSchema, StateInput, UserPreferences
from email_assistant.utils import parse_gmail, format_for_display, format_gmail_markdown
import os
from dotenv import load_dotenv

# Load environment variables from .env file in the same directory as this file
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)
else:
    # Fallback to current directory for standard usage
    load_dotenv()

tools = get_tools(
    [
        "send_email_tool",
        "schedule_meeting_tool",
        "check_calendar_tool",
        "Question",
        "Done",
    ],
    include_gmail=True,
)
tools_by_name = get_tools_by_name(tools)

llm = init_chat_model("openai:gpt-4.1", temperature=0.0)
llm_router = llm.with_structured_output(RouterSchema)

llm = init_chat_model("openai:gpt-4.1", temperature=0.0)
llm_with_tools = llm.bind_tools(tools, tool_choice="required")


def get_memory(store, namespace, default_content=None):
    """Get memory from the store or initialize with default if it doesn't exist.

    Args:
        store: LangGraph BaseStore instance to search for existing memory
        namespace: Tuple defining the memory namespace, e.g. ("email_assistant", "triage_preferences")
        default_content: Default content to use if memory doesn't exist

    Returns:
        str: The content of the memory profile, either from existing memory or the default
    """

    user_preferences = store.get(namespace, "user_preferences")

    if user_preferences:
        return user_preferences.value

    else:
        store.put(namespace, "user_preferences", default_content)
        user_preferences = default_content

    # Return the default content
    return user_preferences


def update_memory(store, namespace, messages):
    """Update memory profile in the store.

    Args:
        store: LangGraph BaseStore instance to update memory
        namespace: Tuple defining the memory namespace, e.g. ("email_assistant", "triage_preferences")
        messages: List of messages to update the memory with
    """

    user_preferences = store.get(namespace, "user_preferences")
    llm = init_chat_model("openai:gpt-4.1", temperature=0.0).with_structured_output(
        UserPreferences
    )
    result = llm.invoke(
        [
            {
                "role": "system",
                "content": MEMORY_UPDATE_INSTRUCTIONS.format(
                    current_profile=user_preferences.value, namespace=namespace
                ),
            },
        ]
        + messages
    )
    store.put(namespace, "user_preferences", result.user_preferences)


# Nodes
def triage_router(
    state: State, store: BaseStore
) -> Command[Literal["triage_interrupt_handler", "response_agent", "__end__"]]:
    """Analyze email content to decide if we should respond, notify, or ignore.

    The triage step prevents the assistant from wasting time on:
    - Marketing emails and spam
    - Company-wide announcements
    - Messages meant for other teams
    """

    # Parse the email input
    author, to, subject, email_thread, email_id = parse_gmail(state["email_input"])
    user_prompt = triage_user_prompt.format(
        author=author, to=to, subject=subject, email_thread=email_thread
    )

    email_markdown = format_gmail_markdown(subject, author, to, email_thread, email_id)

    triage_instructions = get_memory(
        store, ("email_assistant", "triage_preferences"), default_triage_instructions
    )

    system_prompt = triage_system_prompt.format(
        background=default_background,
        triage_instructions=triage_instructions,
    )
    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    # Decision
    classification = result.classification

    # Process the classification decision
    if classification == "respond":
        print("📧 Classification: RESPOND - This email requires a response")
        # Next node
        goto = "response_agent"
        # Update the state
        update = {
            "classification_decision": result.classification,
            "messages": [
                {"role": "user", "content": f"Respond to the email: {email_markdown}"}
            ],
        }

    elif classification == "ignore":
        print("🚫 Classification: IGNORE - This email can be safely ignored")

        # Next node
        goto = END
        # Update the state
        update = {
            "classification_decision": classification,
        }

    elif classification == "notify":
        print("🔔 Classification: NOTIFY - This email contains important information")

        # Next node
        goto = "triage_interrupt_handler"
        # Update the state
        update = {
            "classification_decision": classification,
        }

    else:
        raise ValueError(f"Invalid classification: {classification}")

    return Command(goto=goto, update=update)


def triage_interrupt_handler(
    state: State, store: BaseStore
) -> Command[Literal["response_agent", "__end__"]]:
    """Handles interrupts from the triage step"""

    author, to, subject, email_thread, email_id = parse_gmail(state["email_input"])

    email_markdown = format_gmail_markdown(subject, author, to, email_thread, email_id)

    messages = [
        {"role": "user", "content": f"Email to notify user about: {email_markdown}"}
    ]

    request = {
        "action_request": {
            "action": f"Email Assistant: {state['classification_decision']}",
            "args": {},
        },
        "config": {
            "allow_ignore": True,
            "allow_respond": True,
            "allow_edit": False,
            "allow_accept": False,
        },
        # Email to show in Agent Inbox
        "description": email_markdown,
    }

    response = interrupt([request])[0]

    if response["type"] == "response":
        user_input = response["args"]
        messages.append(
            {
                "role": "user",
                "content": f"User wants to reply to the email. Use this feedback to respond: {user_input}",
            }
        )
        update_memory(
            store,
            ("email_assistant", "triage_preferences"),
            [
                {
                    "role": "user",
                    "content": f"The user decided to respond to the email, so update the triage preferences to capture this.",
                }
            ]
            + messages,
        )

        goto = "response_agent"

    elif response["type"] == "ignore":
        messages.append(
            {
                "role": "user",
                "content": f"The user decided to ignore the email even though it was classified as notify. Update triage preferences to capture this.",
            }
        )
        update_memory(store, ("email_assistant", "triage_preferences"), messages)
        goto = END

    # Catch all other responses
    else:
        raise ValueError(f"Invalid response: {response}")

    # Update the state
    update = {
        "messages": messages,
    }

    return Command(goto=goto, update=update)


def llm_call(state: State, store: BaseStore):
    """LLM decides whether to call a tool or not"""

    cal_preferences = get_memory(
        store, ("email_assistant", "cal_preferences"), default_cal_preferences
    )

    response_preferences = get_memory(
        store, ("email_assistant", "response_preferences"), default_response_preferences
    )

    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    {
                        "role": "system",
                        "content": agent_system_prompt_hitl_memory.format(
                            tools_prompt=GMAIL_TOOLS_PROMPT,
                            background=default_background,
                            response_preferences=response_preferences,
                            cal_preferences=cal_preferences,
                        ),
                    }
                ]
                + state["messages"]
            )
        ]
    }


def interrupt_handler(
    state: State, store: BaseStore
) -> Command[Literal["llm_call", "__end__"]]:
    """Creates an interrupt for human review of tool calls"""

    result = []

    goto = "llm_call"

    for tool_call in state["messages"][-1].tool_calls:
        hitl_tools = ["send_email_tool", "schedule_meeting_tool", "Question"]

        if tool_call["name"] not in hitl_tools:
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append(
                {
                    "role": "tool",
                    "content": observation,
                    "tool_call_id": tool_call["id"],
                }
            )
            continue

        email_input = state["email_input"]
        author, to, subject, email_thread, email_id = parse_gmail(email_input)
        original_email_markdown = format_gmail_markdown(
            subject, author, to, email_thread, email_id
        )

        tool_display = format_for_display(tool_call)
        description = original_email_markdown + tool_display

        # Configure what actions are allowed in Agent Inbox
        if tool_call["name"] == "send_email_tool":
            config = {
                "allow_ignore": True,
                "allow_respond": True,
                "allow_edit": True,
                "allow_accept": True,
            }
        elif tool_call["name"] == "schedule_meeting_tool":
            config = {
                "allow_ignore": True,
                "allow_respond": True,
                "allow_edit": True,
                "allow_accept": True,
            }
        elif tool_call["name"] == "Question":
            config = {
                "allow_ignore": True,
                "allow_respond": True,
                "allow_edit": False,
                "allow_accept": False,
            }
        else:
            raise ValueError(f"Invalid tool call: {tool_call['name']}")

        request = {
            "action_request": {"action": tool_call["name"], "args": tool_call["args"]},
            "config": config,
            "description": description,
        }

        response = interrupt([request])[0]

        if response["type"] == "accept":
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append(
                {
                    "role": "tool",
                    "content": observation,
                    "tool_call_id": tool_call["id"],
                }
            )

        elif response["type"] == "edit":
            tool = tools_by_name[tool_call["name"]]
            initial_tool_call = tool_call["args"]

            edited_args = response["args"]["args"]
            ai_message = state["messages"][
                -1
            ]  # Get the most recent message from the state
            current_id = tool_call["id"]  # Store the ID of the tool call being edited

            updated_tool_calls = [
                tc for tc in ai_message.tool_calls if tc["id"] != current_id
            ] + [
                {
                    "type": "tool_call",
                    "name": tool_call["name"],
                    "args": edited_args,
                    "id": current_id,
                }
            ]

            result.append(
                ai_message.model_copy(update={"tool_calls": updated_tool_calls})
            )

            if tool_call["name"] == "send_email_tool":
                observation = tool.invoke(edited_args)

                result.append(
                    {"role": "tool", "content": observation, "tool_call_id": current_id}
                )

                update_memory(
                    store,
                    ("email_assistant", "response_preferences"),
                    [
                        {
                            "role": "user",
                            "content": f"User edited the email response. Here is the initial email generated by the assistant: {initial_tool_call}. Here is the edited email: {edited_args}. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.",
                        }
                    ],
                )

            # Save feedback in memory and update the schedule_meeting tool call with the edited content from Agent Inbox
            elif tool_call["name"] == "schedule_meeting_tool":
                # Execute the tool with edited args
                observation = tool.invoke(edited_args)

                # Add only the tool response message
                result.append(
                    {"role": "tool", "content": observation, "tool_call_id": current_id}
                )

                # This is new: update the memory
                update_memory(
                    store,
                    ("email_assistant", "cal_preferences"),
                    [
                        {
                            "role": "user",
                            "content": f"User edited the calendar invitation. Here is the initial calendar invitation generated by the assistant: {initial_tool_call}. Here is the edited calendar invitation: {edited_args}. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.",
                        }
                    ],
                )

            else:
                raise ValueError(f"Invalid tool call: {tool_call['name']}")

        elif response["type"] == "ignore":
            if tool_call["name"] == "send_email_tool":
                result.append(
                    {
                        "role": "tool",
                        "content": "User ignored this email draft. Ignore this email and end the workflow.",
                        "tool_call_id": tool_call["id"],
                    }
                )
                goto = END
                update_memory(
                    store,
                    ("email_assistant", "triage_preferences"),
                    state["messages"]
                    + result
                    + [
                        {
                            "role": "user",
                            "content": f"The user ignored the email draft. That means they did not want to respond to the email. Update the triage preferences to ensure emails of this type are not classified as respond. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.",
                        }
                    ],
                )

            elif tool_call["name"] == "schedule_meeting_tool":
                result.append(
                    {
                        "role": "tool",
                        "content": "User ignored this calendar meeting draft. Ignore this email and end the workflow.",
                        "tool_call_id": tool_call["id"],
                    }
                )
                goto = END
                update_memory(
                    store,
                    ("email_assistant", "triage_preferences"),
                    state["messages"]
                    + result
                    + [
                        {
                            "role": "user",
                            "content": f"The user ignored the calendar meeting draft. That means they did not want to schedule a meeting for this email. Update the triage preferences to ensure emails of this type are not classified as respond. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.",
                        }
                    ],
                )

            elif tool_call["name"] == "Question":
                result.append(
                    {
                        "role": "tool",
                        "content": "User ignored this question. Ignore this email and end the workflow.",
                        "tool_call_id": tool_call["id"],
                    }
                )
                goto = END
                update_memory(
                    store,
                    ("email_assistant", "triage_preferences"),
                    state["messages"]
                    + result
                    + [
                        {
                            "role": "user",
                            "content": f"The user ignored the Question. That means they did not want to answer the question or deal with this email. Update the triage preferences to ensure emails of this type are not classified as respond. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.",
                        }
                    ],
                )

            else:
                raise ValueError(f"Invalid tool call: {tool_call['name']}")

        elif response["type"] == "response":
            # User provided feedback
            user_feedback = response["args"]
            if tool_call["name"] == "send_email_tool":
                # Don't execute the tool, and add a message with the user feedback to incorporate into the email
                result.append(
                    {
                        "role": "tool",
                        "content": f"User gave feedback, which can we incorporate into the email. Feedback: {user_feedback}",
                        "tool_call_id": tool_call["id"],
                    }
                )
                # This is new: update the memory
                update_memory(
                    store,
                    ("email_assistant", "response_preferences"),
                    state["messages"]
                    + result
                    + [
                        {
                            "role": "user",
                            "content": f"User gave feedback, which we can use to update the response preferences. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.",
                        }
                    ],
                )

            elif tool_call["name"] == "schedule_meeting_tool":
                # Don't execute the tool, and add a message with the user feedback to incorporate into the email
                result.append(
                    {
                        "role": "tool",
                        "content": f"User gave feedback, which can we incorporate into the meeting request. Feedback: {user_feedback}",
                        "tool_call_id": tool_call["id"],
                    }
                )
                # This is new: update the memory
                update_memory(
                    store,
                    ("email_assistant", "cal_preferences"),
                    state["messages"]
                    + result
                    + [
                        {
                            "role": "user",
                            "content": f"User gave feedback, which we can use to update the calendar preferences. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.",
                        }
                    ],
                )

            elif tool_call["name"] == "Question":
                # Don't execute the tool, and add a message with the user feedback to incorporate into the email
                result.append(
                    {
                        "role": "tool",
                        "content": f"User answered the question, which can we can use for any follow up actions. Feedback: {user_feedback}",
                        "tool_call_id": tool_call["id"],
                    }
                )

            else:
                raise ValueError(f"Invalid tool call: {tool_call['name']}")

    # Update the state
    update = {
        "messages": result,
    }

    return Command(goto=goto, update=update)


# Conditional edge function
def should_continue(
    state: State, store: BaseStore
) -> Literal["interrupt_handler", "mark_as_read_node"]:
    """Route to tool handler, or end if Done tool called"""
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "Done":
                # TODO: Here, we could update the background memory with the email-response for follow up actions.
                return "mark_as_read_node"
            else:
                return "interrupt_handler"


def mark_as_read_node(state: State):
    email_input = state["email_input"]
    author, to, subject, email_thread, email_id = parse_gmail(email_input)
    mark_as_read(email_id)


agent_builder = StateGraph(State)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("interrupt_handler", interrupt_handler)
agent_builder.add_node("mark_as_read_node", mark_as_read_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "interrupt_handler": "interrupt_handler",
        "mark_as_read_node": "mark_as_read_node",
    },
)
agent_builder.add_edge("mark_as_read_node", END)

response_agent = agent_builder.compile()

overall_workflow = (
    StateGraph(State, input_schema=StateInput)
    .add_node(triage_router)
    .add_node(triage_interrupt_handler)
    .add_node("response_agent", response_agent)
    .add_node("mark_as_read_node", mark_as_read_node)
    .add_edge(START, "triage_router")
    .add_edge("mark_as_read_node", END)
)

email_assistant = overall_workflow.compile()

if __name__ == "__main__":
    print("Email Assistant Graph loaded.")
    print("To run locally, use the LangGraph CLI:")
    print("  langgraph dev")
    print("\nOr invoke it programmatically in a script.")
