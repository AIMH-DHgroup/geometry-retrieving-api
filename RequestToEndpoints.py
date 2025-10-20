import requests
import json
import re
from odf.opendocument import load
from odf.table import Table, TableRow, TableCell
from odf.text import P
import os

def read_ods_texts(file_path):
    """Reads all text cells from an ODS file and returns them as a list of strings."""
    doc = load(file_path)
    texts = []
    for sheet in doc.getElementsByType(Table):
        for row in sheet.getElementsByType(TableRow):
            row_text = []
            for cell in row.getElementsByType(TableCell):
                cell_text = ""
                for p in cell.getElementsByType(P):
                    para_text = ""
                    for n in p.childNodes:
                        if hasattr(n, "data"):
                            para_text += n.data
                    cell_text += para_text
                if cell_text.strip():
                    row_text.append(cell_text.strip())
            if row_text:
                texts.append(" ".join(row_text))
    return texts


def sanitize_filename(url):
    """Creates a valid filename from a URL."""
    name = re.sub(r"https?://", "", url)
    name = re.sub(r"[^A-Za-z0-9_-]", "_", name)
    return name


def build_excel_xml(texts):
    """Builds an Excel 2003-style XML file (SpreadsheetML) from a list of text strings."""
    rows_xml = []
    for text in texts:
        escaped_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        row_xml = (
            "    <Row>\n"
            f"      <Cell>\n"
            f"        <Data ss:Type=\"String\">{escaped_text}</Data>\n"
            f"      </Cell>\n"
            "    </Row>\n"
        )
        rows_xml.append(row_xml)

    xml_content = (
        '<?xml version="1.0"?>\n'
        '<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet" '
        'xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">\n'
        '  <Worksheet ss:Name="Foglio1">\n'
        '    <Table>\n'
        + "".join(rows_xml) +
        '    </Table>\n'
        '  </Worksheet>\n'
        '</Workbook>'
    )
    return xml_content


def main():
    endpoint = input("Enter the endpoint URL: ").strip()
    ods_path = input("Enter the path to the .ods file: ").strip()

    if not os.path.exists(ods_path):
        print("❌ Error: ODS file not found.")
        return

    # Convert HTTPS to HTTP for localhost if needed
    if endpoint.startswith("https://127.0.0.1") or endpoint.startswith("https://localhost"):
        print("⚠️ Switching to HTTP (local server usually not HTTPS).")
        endpoint = endpoint.replace("https://", "http://")

    print("📖 Reading ODS file...")
    texts = read_ods_texts(ods_path)

    if not texts:
        print("⚠️ No text found in the ODS file.")
        return

    print(f"🧩 Building XML with {len(texts)} rows...")
    xml_content = build_excel_xml(texts)

    print(f"📤 Sending XML to endpoint: {endpoint}")
    try:
        response = requests.post(
            endpoint,
            files={"file": ("events.xml", xml_content.encode("utf-8"), "application/xml")},
            timeout=300
        )

        try:
            data = response.json()
        except Exception:
            data = {"status": response.status_code, "text": response.text}

    except Exception as e:
        print(f"⚠️ Error sending request: {e}")
        data = {"error": str(e)}

    output_filename = sanitize_filename(endpoint) + ".json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ File saved as: {output_filename}")


if __name__ == "__main__":
    main()
