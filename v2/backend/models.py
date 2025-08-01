from pydantic import BaseModel, EmailStr
from typing import Literal, Optional

class UserAuth(BaseModel):
    email: EmailStr
    password: str
    action: Literal["signup", "login"] 
    name: Optional[str] = None
    model: Optional[str] = "1_0"