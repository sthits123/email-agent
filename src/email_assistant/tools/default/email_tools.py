from typing import Optional
from langchain_core.tools import tool
from pydantic import BaseModel


@tool
def write_email(to: str, content: str, subject: str) -> str:
    """Draft an email to send to a recipient.

    Args:
        to: Email address of the recipient
        subject: Subject line of the email
        content: Body content of the email
    """
    return f"Email drafted to {to} with subject: {subject}"


@tool
def Done() -> str:
    """Indicate that the task has been completed. Use this when you have finished processing the email."""
    return "Task completed"


@tool
def Question(question: str) -> str:
    """Ask the user a question when you need clarification or additional information.

    Args:
        question: The question to ask the user
    """
    return f"Question asked: {question}"


@tool
def triage_email(email_content: str) -> str:
    """Triage an email to determine its category and required action.

    Args:
        email_content: The content of the email to triage
    """
    return "Email triaged as general, action: respond"
