"""store expected relevance labels with evaluation questions

Revision ID: 9ac4750c1e7f
Revises: 54e41a1d3f0b
Create Date: 2026-10-01
"""

from alembic import op
import sqlalchemy as sa


revision = "9ac4750c1e7f"
down_revision = "54e41a1d3f0b"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("evaluation_questions", sa.Column("relevant_chunk_ids", sa.JSON(), nullable=False, server_default="[]"))
    op.add_column("evaluation_questions", sa.Column("relevant_contexts", sa.JSON(), nullable=False, server_default="[]"))
    op.add_column("evaluation_questions", sa.Column("expected_document", sa.String(length=500), nullable=True))
    op.add_column("evaluation_questions", sa.Column("expected_page", sa.Integer(), nullable=True))
    op.add_column("evaluation_questions", sa.Column("expected_chunk", sa.String(length=100), nullable=True))


def downgrade() -> None:
    op.drop_column("evaluation_questions", "expected_chunk")
    op.drop_column("evaluation_questions", "expected_page")
    op.drop_column("evaluation_questions", "expected_document")
    op.drop_column("evaluation_questions", "relevant_contexts")
    op.drop_column("evaluation_questions", "relevant_chunk_ids")
