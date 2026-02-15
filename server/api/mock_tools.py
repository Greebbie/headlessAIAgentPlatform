"""Mock tool endpoints for development and testing.

These provide real, working tool endpoints that the agent can call
via function calling. Useful for testing the full tool-calling pipeline.
"""

from __future__ import annotations

import math
from datetime import datetime

from fastapi import APIRouter

router = APIRouter()


@router.post("/calculator")
async def calculator(data: dict):
    """A basic calculator tool. Supports +, -, *, /, sqrt, pow, log."""
    expression = data.get("expression", "")
    op = data.get("operation", "")
    a = data.get("a")
    b = data.get("b")

    # Mode 1: structured operation
    if op:
        try:
            a = float(a) if a is not None else 0
            b = float(b) if b is not None else 0
            if op in ("add", "+"):
                result = a + b
            elif op in ("subtract", "-"):
                result = a - b
            elif op in ("multiply", "*", "×"):
                result = a * b
            elif op in ("divide", "/", "÷"):
                if b == 0:
                    return {"error": "Division by zero", "success": False}
                result = a / b
            elif op in ("sqrt", "square_root"):
                if a < 0:
                    return {"error": "Cannot sqrt negative number", "success": False}
                result = math.sqrt(a)
            elif op in ("pow", "power", "**"):
                result = math.pow(a, b)
            elif op in ("log", "ln"):
                if a <= 0:
                    return {"error": "Cannot log non-positive number", "success": False}
                result = math.log(a) if b == 0 else math.log(a, b)
            elif op in ("mod", "%"):
                if b == 0:
                    return {"error": "Division by zero", "success": False}
                result = a % b
            else:
                return {"error": f"Unknown operation: {op}", "success": False}

            # Format nicely
            if result == int(result):
                result = int(result)
            return {"result": result, "operation": op, "a": a, "b": b, "success": True}
        except Exception as e:
            return {"error": str(e), "success": False}

    # Mode 2: expression string (basic safe eval)
    if expression:
        try:
            # Only allow safe math operations
            allowed = set("0123456789+-*/.() ")
            if not all(c in allowed for c in expression):
                return {"error": "Invalid characters in expression", "success": False}
            result = eval(expression)  # noqa: S307 — restricted to digits+operators
            if isinstance(result, float) and result == int(result):
                result = int(result)
            return {"result": result, "expression": expression, "success": True}
        except Exception as e:
            return {"error": f"Failed to evaluate: {e}", "success": False}

    return {"error": "Provide 'operation' + 'a'/'b', or 'expression'", "success": False}


@router.post("/weather")
async def weather(data: dict):
    """Mock weather tool — returns simulated weather for any city."""
    city = data.get("city", "").strip()
    if not city:
        return {"error": "Please provide a city name", "success": False}

    # Simulated weather data
    import hashlib
    seed = int(hashlib.md5(city.encode()).hexdigest()[:8], 16)
    temp = 15 + (seed % 25)  # 15-39°C
    humidity = 40 + (seed % 50)  # 40-89%
    conditions = ["晴", "多云", "阴", "小雨", "大雨", "雪"][seed % 6]

    return {
        "city": city,
        "temperature": temp,
        "humidity": humidity,
        "condition": conditions,
        "wind": f"{(seed % 6) + 1}级",
        "forecast": f"{city}今日{conditions}，气温{temp}°C，湿度{humidity}%",
        "updated_at": datetime.now().isoformat(),
        "success": True,
    }


@router.post("/unit_converter")
async def unit_converter(data: dict):
    """Unit converter tool — converts between common units."""
    value = data.get("value")
    from_unit = data.get("from_unit", "").lower()
    to_unit = data.get("to_unit", "").lower()

    if value is None:
        return {"error": "Please provide a value to convert", "success": False}

    try:
        value = float(value)
    except (ValueError, TypeError):
        return {"error": f"Invalid value: {value}", "success": False}

    # Conversion tables
    length = {
        ("km", "m"): 1000, ("m", "km"): 0.001,
        ("m", "cm"): 100, ("cm", "m"): 0.01,
        ("km", "mile"): 0.621371, ("mile", "km"): 1.60934,
        ("m", "ft"): 3.28084, ("ft", "m"): 0.3048,
        ("inch", "cm"): 2.54, ("cm", "inch"): 0.393701,
    }
    weight = {
        ("kg", "g"): 1000, ("g", "kg"): 0.001,
        ("kg", "lb"): 2.20462, ("lb", "kg"): 0.453592,
        ("kg", "oz"): 35.274, ("oz", "kg"): 0.0283495,
    }
    temperature = {}  # handled specially

    key = (from_unit, to_unit)

    # Temperature special handling
    if from_unit in ("c", "celsius", "℃") and to_unit in ("f", "fahrenheit", "℉"):
        result = value * 9 / 5 + 32
    elif from_unit in ("f", "fahrenheit", "℉") and to_unit in ("c", "celsius", "℃"):
        result = (value - 32) * 5 / 9
    elif key in length:
        result = value * length[key]
    elif key in weight:
        result = value * weight[key]
    elif from_unit == to_unit:
        result = value
    else:
        return {"error": f"Unsupported conversion: {from_unit} → {to_unit}", "success": False}

    result = round(result, 4)
    return {
        "original_value": value,
        "from_unit": from_unit,
        "to_unit": to_unit,
        "result": result,
        "formatted": f"{value} {from_unit} = {result} {to_unit}",
        "success": True,
    }


@router.post("/timestamp")
async def timestamp_tool(data: dict):
    """Current timestamp / date-time tool."""
    fmt = data.get("format", "iso")
    tz_name = data.get("timezone", "UTC")

    now = datetime.now()
    if fmt == "iso":
        formatted = now.isoformat()
    elif fmt == "date":
        formatted = now.strftime("%Y-%m-%d")
    elif fmt == "time":
        formatted = now.strftime("%H:%M:%S")
    elif fmt == "unix":
        formatted = str(int(now.timestamp()))
    else:
        formatted = now.strftime(fmt)

    return {
        "datetime": formatted,
        "timezone": tz_name,
        "unix_timestamp": int(now.timestamp()),
        "success": True,
    }
