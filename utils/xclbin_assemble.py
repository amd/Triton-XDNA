#!/usr/bin/env python3
"""
Minimal xclbin assembler for Triton-XDNA on Windows.

Replaces xclbinutil for the specific use case of assembling an xclbin
from the JSON metadata + PDI artifacts produced by aircc/aiecc.

Usage:
    python xclbin_assemble.py --mem-topology mem.json --kernels kern.json \
                              --aie-partition part.json --output aie.xclbin

The xclbin binary format (axlf) is defined in xrt/detail/xclbin.h.
Struct sizes verified by static_asserts in the header:
  axlf_section_header = 40     (4-byte padding between m_sectionName and m_sectionOffset)
  axlf_header         = 152
  mem_data             = 40
  connection           = 12
  ip_data              = 80
  aie_partition        = 184   (binary header + heap, NOT JSON)
  aie_pdi              = 96
  cdo_group            = 96
  aie_partition_info   = 88
  array_offset         = 8
"""
import argparse
import json
import os
import struct
import time
import uuid as uuid_mod


# ── axlf_section_kind enum values ──
SECTION_EMBEDDED_METADATA = 2
SECTION_MEM_TOPOLOGY = 6
SECTION_CONNECTIVITY = 7
SECTION_IP_LAYOUT = 8
SECTION_AIE_PARTITION = 32

# ── MEM_TYPE enum values ──
MEM_TYPE_MAP = {
    "MEM_DDR3": 0, "MEM_DDR4": 1, "MEM_DRAM": 2, "MEM_STREAMING": 3,
    "MEM_PREALLOCATED_GLOB": 4, "MEM_ARE": 5, "MEM_HBM": 6,
    "MEM_BRAM": 7, "MEM_URAM": 8, "MEM_STREAMING_CONNECTION": 9,
    "MEM_HOST": 10, "MEM_PS_KERNEL": 11,
}

# ── IP_TYPE enum values ──
IP_PS_KERNEL = 7

# ── CDO_Type enum values ──
CDO_TYPE_MAP = {"UNKNOWN": 0, "PRIMARY": 1, "LITE": 2, "PRE_POST": 3}


def _pad_to_8(data: bytes) -> bytes:
    """Pad data to 8-byte alignment."""
    remainder = len(data) % 8
    if remainder:
        data += b'\x00' * (8 - remainder)
    return data


def _pack_section_header(kind: int, name: str, offset: int, size: int) -> bytes:
    """
    Pack a 40-byte axlf_section_header.
    Layout: uint32 m_sectionKind (4) | char m_sectionName[16] (16) |
            4-byte padding | uint64 m_sectionOffset (8) | uint64 m_sectionSize (8)
    Total: 40 bytes.
    """
    name_bytes = name.encode('utf-8')[:16].ljust(16, b'\x00')
    return struct.pack('<I16s4xQQ', kind, name_bytes, offset, size)


def _encode_mem_topology(mem_json: dict) -> bytes:
    """
    Encode MEM_TOPOLOGY section as binary.
    struct mem_topology { int32_t m_count; struct mem_data m_mem_data[]; }
    struct mem_data (40 bytes):
        uint8_t m_type; uint8_t m_used; uint8_t padding[6];
        uint64_t m_size; uint64_t m_base_address; unsigned char m_tag[16];
    """
    topo = mem_json["mem_topology"]
    count = int(topo["m_count"])
    buf = struct.pack('<i', count)
    for md in topo["m_mem_data"]:
        m_type = MEM_TYPE_MAP.get(md["m_type"], 2)
        m_used = int(md["m_used"])
        m_size_kb = int(md["m_sizeKB"], 0)
        m_base = int(md["m_base_address"], 0)
        tag = md["m_tag"].encode('utf-8')[:16].ljust(16, b'\x00')
        buf += struct.pack('<BB6xQQ', m_type, m_used, m_size_kb, m_base)
        buf += tag
    return _pad_to_8(buf)


def _encode_ip_layout(kernels_json: dict) -> bytes:
    """
    Encode IP_LAYOUT section from kernels JSON.
    struct ip_layout { int32_t m_count; struct ip_data m_ip_data[]; }
    struct ip_data (80 bytes):
        uint32_t m_type; uint32_t properties; uint64_t m_base_address;
        uint8_t m_name[64];

    ps_kernel properties bitfield:
        bits [1:0]   = m_subtype  (ST_DPU=1)
        bits [5:4]   = m_functional (FC_DPU=0)
        bits [27:16] = m_kernel_id
    """
    kernels = kernels_json.get("ps-kernels", {}).get("kernels", [])
    instances = []
    for k in kernels:
        kernel_name = k.get("name", "MLIR_AIE")
        ext = k.get("extended-data", {})
        kernel_id = int(ext.get("dpu_kernel_id", "0x901"), 0) & 0xFFF
        subtype = 1 if ext.get("subtype", "DPU") == "DPU" else 0
        functional = int(ext.get("functional", "0"))
        for inst in k.get("instances", []):
            inst_name = inst["name"]
            # ip_data name format: "kernel_name:instance_name"
            cu_name = f"{kernel_name}:{inst_name}"
            instances.append((cu_name, kernel_id, subtype, functional))

    count = len(instances)
    buf = struct.pack('<i', count)
    for cu_name, kernel_id, subtype, functional in instances:
        m_type = IP_PS_KERNEL
        properties = (subtype & 0x3) | ((functional & 0x3) << 4) | ((kernel_id & 0xFFF) << 16)
        # PS kernels use base_address = (uint64_t)-1 meaning "not_used"
        # This is how xclbinutil encodes PS kernel CUs
        m_base = 0xFFFFFFFFFFFFFFFF
        m_name = cu_name.encode('utf-8')[:64].ljust(64, b'\x00')
        buf += struct.pack('<IIQ', m_type, properties, m_base)
        buf += m_name
    return _pad_to_8(buf)


def _encode_connectivity(kernels_json: dict, mem_json: dict) -> bytes:
    """
    Encode CONNECTIVITY section linking kernel args to memory banks.
    struct connectivity { int32_t m_count; struct connection m_connection[]; }
    struct connection (12 bytes): int32_t arg_index, m_ip_layout_index, mem_data_index;
    """
    kernels = kernels_json.get("ps-kernels", {}).get("kernels", [])
    topo = mem_json["mem_topology"]
    mem_data = topo["m_mem_data"]

    tag_to_idx = {}
    for i, md in enumerate(mem_data):
        tag_to_idx[md["m_tag"]] = i

    connections = []
    ip_idx = 0
    for k in kernels:
        for _inst in k.get("instances", []):
            for arg_i, arg in enumerate(k.get("arguments", [])):
                if arg.get("address-qualifier") != "GLOBAL":
                    continue
                mem_conn = arg.get("memory-connection", "HOST")
                mem_idx = tag_to_idx.get(mem_conn, 0)
                connections.append((arg_i, ip_idx, mem_idx))
            ip_idx += 1

    count = len(connections)
    buf = struct.pack('<i', count)
    for arg_idx, ip_layout_idx, mem_data_idx in connections:
        buf += struct.pack('<iii', arg_idx, ip_layout_idx, mem_data_idx)
    return _pad_to_8(buf)


def _encode_embedded_metadata(kernels_json: dict) -> bytes:
    """
    Encode EMBEDDED_METADATA section as XML.
    Required <arg> attributes: name, addressQualifier, id, offset, size, hostSize.
    Optional: type, port, hostOffset.
    """
    TYPE_SIZES = {
        "uint64_t": "0x8", "uint32_t": "0x4", "uint16_t": "0x2", "uint8_t": "0x1",
        "int64_t": "0x8", "int32_t": "0x4", "int16_t": "0x2", "int8_t": "0x1",
        "char *": "0x8", "void*": "0x8", "float*": "0x8", "double*": "0x8",
    }
    kernels = kernels_json.get("ps-kernels", {}).get("kernels", [])
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
    xml += '<project name="Triton-XDNA">\n'
    xml += '  <platform name="xilinx">\n'
    xml += '    <device name="fpga0">\n'
    xml += '      <core name="OCL_REGION_0">\n'
    for k in kernels:
        kname = k.get("name", "MLIR_AIE")
        ktype = k.get("type", "dpu")
        xml += f'        <kernel name="{kname}" language="c" type="{ktype}">\n'
        # Extended data for DPU kernels
        ext = k.get("extended-data", {})
        if ext:
            subtype_val = "1" if ext.get("subtype", "DPU") == "DPU" else "0"
            functional_val = ext.get("functional", "0")
            dpu_kid = ext.get("dpu_kernel_id", "0x901")
            xml += f'          <extended-data subtype="{subtype_val}" '
            xml += f'functional="{functional_val}" dpu_kernel_id="{dpu_kid}"/>\n'
        for ai, arg in enumerate(k.get("arguments", [])):
            aname = arg["name"]
            aq = "0" if arg["address-qualifier"] == "SCALAR" else "1"
            atype = arg.get("type", "void*")
            offset = arg.get("offset", "0x00")
            size = TYPE_SIZES.get(atype, "0x8")
            host_size = size
            xml += f'          <arg name="{aname}" addressQualifier="{aq}" '
            xml += f'id="{ai}" offset="{offset}" size="{size}" '
            xml += f'hostOffset="0x0" hostSize="{host_size}" type="{atype}"'
            if aq == "1":
                xml += ' port="S_AXI_CONTROL"'
            xml += '/>\n'
        for inst in k.get("instances", []):
            xml += f'          <instance name="{inst["name"]}">\n'
            xml += f'            <addrRemap base="0x0"/>\n'
            xml += f'          </instance>\n'
        xml += '        </kernel>\n'
    xml += '      </core>\n'
    xml += '    </device>\n'
    xml += '  </platform>\n'
    xml += '</project>\n'
    return _pad_to_8(xml.encode('utf-8'))


class _SectionHeap:
    """Helper to build a binary heap with 8-byte aligned allocations."""

    def __init__(self, base_offset: int):
        self.base = base_offset
        self.data = bytearray()

    def alloc(self, raw: bytes, align: bool = True) -> int:
        """Write data to heap, return offset from section start.
        Pads to 8-byte alignment after write if align=True."""
        offset = self.base + len(self.data)
        self.data.extend(raw)
        if align:
            remainder = len(self.data) % 8
            if remainder:
                self.data.extend(b'\x00' * (8 - remainder))
        return offset

    def alloc_string(self, s: str) -> int:
        """Write null-terminated string to heap, return offset."""
        return self.alloc(s.encode('utf-8') + b'\x00')


def _encode_aie_partition(partition_json: dict, base_dir: str) -> bytes:
    """
    Encode AIE_PARTITION section in the custom binary format expected by XRT.

    Binary layout:
        aie_partition header  (184 bytes)
        heap data             (variable, contains strings, PDI data, structs)

    All offsets in the header/sub-structs are byte offsets from the start
    of this section.
    """
    part = partition_json["aie_partition"]

    AIE_PARTITION_HDR_SIZE = 184
    heap = _SectionHeap(AIE_PARTITION_HDR_SIZE)

    # ── Name string ──
    name = part.get("name", "QoS")
    name_offset = heap.alloc_string(name)

    # ── Kernel commit ID string ──
    commit_id = part.get("kernel_commit_id", "")
    commit_offset = heap.alloc_string(commit_id) if commit_id else 0

    # ── Scalars ──
    ops_per_cycle = int(part.get("operations_per_cycle", "2048"))
    inference_fp = int(part.get("inference_fingerprint", "0"))
    pre_post_fp = int(part.get("pre_post_fingerprint", "0"))

    # ── Partition info ──
    partition_info = part.get("partition", {})
    column_width = int(partition_info.get("column_width", 4))
    start_cols = partition_info.get("start_columns", [])

    if start_cols:
        cols_data = b''.join(struct.pack('<H', c) for c in start_cols)
        start_cols_offset = heap.alloc(cols_data)
        start_cols_count = len(start_cols)
    else:
        start_cols_offset = 0
        start_cols_count = 0

    # ── Process PDIs ──
    pdis = part.get("PDIs", [])
    aie_pdi_structs = []

    for pdi_entry in pdis:
        # UUID
        uuid_str = pdi_entry.get("uuid", "")
        if uuid_str:
            pdi_uuid_bytes = uuid_mod.UUID(uuid_str).bytes  # 16 bytes, RFC 4122
        else:
            pdi_uuid_bytes = b'\x00' * 16

        # Read raw PDI file
        pdi_filename = pdi_entry.get("file_name", "")
        pdi_path = pdi_filename
        if not os.path.isabs(pdi_path):
            pdi_path = os.path.join(base_dir, pdi_filename)
        if pdi_path and os.path.exists(pdi_path):
            with open(pdi_path, 'rb') as f:
                pdi_raw = f.read()
            pdi_image_offset = heap.alloc(pdi_raw)
            pdi_image_count = len(pdi_raw)
        else:
            pdi_image_offset = 0
            pdi_image_count = 0

        # Process CDO groups
        cdo_entries = pdi_entry.get("cdo_groups", [])
        cdo_group_structs = []

        for cg in cdo_entries:
            cg_name = cg.get("name", "DPU")
            cg_name_offset = heap.alloc_string(cg_name)

            cdo_type = CDO_TYPE_MAP.get(cg.get("type", "PRIMARY"), 1)
            pdi_id = int(cg.get("pdi_id", "0x01"), 0)

            # dpu_kernel_ids: array of uint64_t
            kernel_ids = cg.get("dpu_kernel_ids", [])
            if kernel_ids:
                kids_data = b''.join(
                    struct.pack('<Q', int(k, 0) if isinstance(k, str) else k)
                    for k in kernel_ids
                )
                kids_offset = heap.alloc(kids_data)
                kids_count = len(kernel_ids)
            else:
                kids_offset = 0
                kids_count = 0

            # pre_cdo_groups: array of uint32_t
            pre_groups = cg.get("pre_cdo_groups", [])
            if pre_groups:
                pg_data = b''.join(
                    struct.pack('<I', int(p, 0) if isinstance(p, str) else p)
                    for p in pre_groups
                )
                pg_offset = heap.alloc(pg_data)
                pg_count = len(pre_groups)
            else:
                pg_offset = 0
                pg_count = 0

            # cdo_group struct (96 bytes)
            cdo_s = struct.pack('<I', cg_name_offset)        # mpo_name       (4)
            cdo_s += struct.pack('<B3x', cdo_type)           # cdo_type+pad   (4)
            cdo_s += struct.pack('<Q', pdi_id)               # pdi_id         (8)
            cdo_s += struct.pack('<II', kids_count, kids_offset)   # dpu_kernel_ids (8)
            cdo_s += struct.pack('<II', pg_count, pg_offset)       # pre_cdo_groups (8)
            cdo_s += b'\x00' * 64                            # reserved       (64)
            assert len(cdo_s) == 96, f"cdo_group size {len(cdo_s)} != 96"
            cdo_group_structs.append(cdo_s)

        # Write all cdo_group structs to heap
        if cdo_group_structs:
            cdo_all = b''.join(cdo_group_structs)
            cdo_groups_offset = heap.alloc(cdo_all)
            cdo_groups_count = len(cdo_group_structs)
        else:
            cdo_groups_offset = 0
            cdo_groups_count = 0

        # aie_pdi struct (96 bytes)
        pdi_s = pdi_uuid_bytes                                          # uuid          (16)
        pdi_s += struct.pack('<II', pdi_image_count, pdi_image_offset)  # pdi_image     (8)
        pdi_s += struct.pack('<II', cdo_groups_count, cdo_groups_offset)# cdo_groups    (8)
        pdi_s += b'\x00' * 64                                          # reserved      (64)
        assert len(pdi_s) == 96, f"aie_pdi size {len(pdi_s)} != 96"
        aie_pdi_structs.append(pdi_s)

    # Write all aie_pdi structs to heap
    if aie_pdi_structs:
        pdi_all = b''.join(aie_pdi_structs)
        aie_pdi_offset = heap.alloc(pdi_all)
        aie_pdi_count = len(aie_pdi_structs)
    else:
        aie_pdi_offset = 0
        aie_pdi_count = 0

    # ── Build aie_partition header (184 bytes) ──
    hdr = struct.pack('<B3x', 0)                            # schema_version + padding0  (4)
    hdr += struct.pack('<I', name_offset)                    # mpo_name                   (4)
    hdr += struct.pack('<I', ops_per_cycle)                  # operations_per_cycle       (4)
    hdr += b'\x00' * 4                                      # padding                    (4)
    hdr += struct.pack('<Q', inference_fp)                   # inference_fingerprint      (8)
    hdr += struct.pack('<Q', pre_post_fp)                    # pre_post_fingerprint       (8)
    # aie_partition_info (88 bytes)
    hdr += struct.pack('<H6x', column_width)                # column_width + padding     (8)
    hdr += struct.pack('<II', start_cols_count, start_cols_offset)  # start_columns       (8)
    hdr += b'\x00' * 72                                     # reserved                   (72)
    # end aie_partition_info
    hdr += struct.pack('<II', aie_pdi_count, aie_pdi_offset)# aie_pdi                    (8)
    hdr += struct.pack('<I', commit_offset)                  # kernel_commit_id           (4)
    hdr += b'\x00' * 52                                     # reserved                   (52)
    assert len(hdr) == AIE_PARTITION_HDR_SIZE, \
        f"aie_partition header size {len(hdr)} != {AIE_PARTITION_HDR_SIZE}"

    return bytes(hdr) + bytes(heap.data)


def assemble_xclbin(mem_topology_json: dict, kernels_json: dict,
                     aie_partition_json: dict, output_path: str):
    """
    Assemble a valid xclbin file from JSON metadata sections.

    xclbin layout:
        axlf preamble  (304 bytes: magic + sig_len + reserved + keyBlock + uniqueId)
        axlf_header    (152 bytes)
        section_headers (40 bytes each)
        section_data    (8-byte aligned)
    """
    base_dir = os.path.dirname(output_path) or "."

    # Encode sections
    mem_topo_data = _encode_mem_topology(mem_topology_json)
    ip_layout_data = _encode_ip_layout(kernels_json)
    connectivity_data = _encode_connectivity(kernels_json, mem_topology_json)
    embedded_meta_data = _encode_embedded_metadata(kernels_json)
    aie_partition_data = _encode_aie_partition(aie_partition_json, base_dir)

    sections = [
        (SECTION_MEM_TOPOLOGY, "mem_topology", mem_topo_data),
        (SECTION_IP_LAYOUT, "ip_layout", ip_layout_data),
        (SECTION_CONNECTIVITY, "", connectivity_data),
        (SECTION_EMBEDDED_METADATA, "", embedded_meta_data),
        (SECTION_AIE_PARTITION, "", aie_partition_data),
    ]

    num_sections = len(sections)

    # ── Sizing ──
    PREAMBLE_SIZE = 304   # magic(8) + sig_len(4) + reserved(28) + keyBlock(256) + uniqueId(8)
    HEADER_SIZE = 152
    SEC_HDR_SIZE = 40

    headers_start = PREAMBLE_SIZE + HEADER_SIZE
    data_start = headers_start + (SEC_HDR_SIZE * num_sections)
    # Align data_start to 8 bytes
    if data_start % 8:
        data_start += 8 - (data_start % 8)

    # Calculate section offsets and total size
    offsets = []
    current = data_start
    for _kind, _name, data in sections:
        offsets.append(current)
        current += len(data)
        # Ensure each section starts 8-byte aligned
        if current % 8:
            current += 8 - (current % 8)

    total_size = current

    # Generate UUID for this xclbin
    xclbin_uuid = uuid_mod.uuid4()

    # ── axlf_header (152 bytes) ──
    header = struct.pack('<QQQ',
                         total_size,           # m_length
                         int(time.time()),     # m_timeStamp
                         0)                    # m_featureRomTimeStamp
    header += struct.pack('<HBB', 0, 2, 1)     # versionPatch=0, Major=2, Minor=1
    header += struct.pack('<HH', 0, 0x1)       # m_mode=FLAT, m_actionMask=AM_LOAD_AIE
    header += b'\x00' * 16                     # m_interface_uuid
    header += b'\x00' * 64                     # m_platformVBNV
    header += xclbin_uuid.bytes                # uuid (16 bytes)
    header += b'\x00' * 16                     # m_debug_bin
    header += struct.pack('<I', num_sections)   # m_numSections
    header += b'\x00' * (HEADER_SIZE - len(header))  # trailing pad to 152

    # ── Preamble (304 bytes) ──
    preamble = b'xclbin2\x00'                  # magic (8)
    preamble += struct.pack('<i', -1)          # sig_length = -1 (none)
    preamble += b'\xFF' * 28                   # reserved
    preamble += b'\x00' * 256                  # keyBlock
    preamble += struct.pack('<Q', int(time.time() * 1000) & 0xFFFFFFFFFFFFFFFF)  # uniqueId
    assert len(preamble) == PREAMBLE_SIZE

    # ── Section headers (40 bytes each, with correct padding) ──
    sec_headers = b''
    for i, (kind, name, data) in enumerate(sections):
        sec_headers += _pack_section_header(kind, name, offsets[i], len(data))

    # Padding between end of headers and start of data
    headers_end = PREAMBLE_SIZE + HEADER_SIZE + len(sec_headers)
    if headers_end < data_start:
        sec_headers += b'\x00' * (data_start - headers_end)

    # ── Write ──
    with open(output_path, 'wb') as f:
        f.write(preamble)
        f.write(header)
        f.write(sec_headers)
        for i, (_kind, _name, data) in enumerate(sections):
            f.write(data)
            # Inter-section padding
            written = len(data)
            if written % 8:
                f.write(b'\x00' * (8 - written % 8))

    actual_size = os.path.getsize(output_path)
    print(f"Generated xclbin: {output_path} ({actual_size} bytes, {num_sections} sections)")


def main():
    parser = argparse.ArgumentParser(description="Minimal xclbin assembler")
    parser.add_argument("--mem-topology", required=True, help="mem_topology JSON file")
    parser.add_argument("--kernels", required=True, help="kernels JSON file")
    parser.add_argument("--aie-partition", required=True, help="aie_partition JSON file")
    parser.add_argument("--output", "-o", required=True, help="Output xclbin file")
    args = parser.parse_args()

    with open(args.mem_topology) as f:
        mem_json = json.load(f)
    with open(args.kernels) as f:
        kern_json = json.load(f)
    with open(args.aie_partition) as f:
        part_json = json.load(f)

    assemble_xclbin(mem_json, kern_json, part_json, args.output)


if __name__ == "__main__":
    main()
