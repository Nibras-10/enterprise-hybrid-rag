"""store generated query answers for history retrieval

Revision ID: 54e41a1d3f0b
Revises: 096d77189e44
Create Date: 2026-10-01
"""

from alembic import op
import sqlalchemy as sa


revision = "54e41a1d3f0b"
down_revision = "096d77189e44"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("queries", sa.Column("answer", sa.Text(), nullable=True))
    op.add_column("queries", sa.Column("citations", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("queries", "citations")
    op.drop_column("queries", "answer")
