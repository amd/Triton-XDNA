// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Reader for the full-ELF kernel binaries aiecc emits (`aiecc --get-full-elf`).
//
// A full ELF carries the PDIs and the control code in one file, along with the
// relocations that say where addresses have to be written into the control
// code. ROCR deliberately knows nothing about any of that: the application
// extracts the pieces, allocates them from the agent's device memory pool,
// patches its own addresses in, and names the buffers in an ordinary dispatch
// packet. The one address it cannot know is the PDI's device address, so it
// passes the offset of that patch site in
// hsa_amd_aie_kernel_dispatch_packet_t::pdi_patch_offset and ROCR fills it in;
// that non-zero offset is also what selects the full-ELF dispatch shape.
//
// Derived from the reference reader in ROCm/rocm-systems#11668
// (rocrtst/suites/aie/aie_full_elf.h), which supports only what its
// vector_scalar_add design needs. Three things here go beyond it, because the
// designs this backend lowers need them:
//
//   * the scratchpad. `scratch-pad-ctrl` is an ordinary relocation symbol whose
//     st_size is the buffer's size; the reference refuses it outright. It is
//     the whole point of reading these ELFs here -- it is how a runtime value
//     (a decode's context length) reaches the device without rewriting the
//     instruction stream per token.
//   * every PDI, not just the patched one. A design that switches
//     configuration with `load_pdi` carries several (the fused decode carries
//     four) and the reference allocates only the one with a patch site.
//   * more than one kernel per file, selected by name.
//
// Reads ELF32 little-endian on a little-endian host, so the on-disk layout is
// the host layout. Bounds-checked throughout: a truncated or hostile file is a
// clean exception, never a read off the end.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <elf.h>

namespace aie_full_elf {

// ELF OS/ABI identifying an aie2p AIE ELF.
inline constexpr std::uint8_t kElfAmdAie2p = 69;

// The relocation symbol naming the control scratchpad. Same name XRT looks for
// (xrt_module.cpp's Control_ScratchPad_Symbol); its st_size is the buffer size.
inline constexpr char kScratchpadSymbol[] = "scratch-pad-ctrl";

// Relocation types, matching the patch schemes the NPU firmware and XRT use.
enum class PatchScheme : std::uint32_t {
  // Fold a buffer address into a shim DMA buffer descriptor. Kernel arguments.
  kShimDma48 = 5,
  // Store a plain 64-bit address. The PDI address and the scratchpad.
  kAddress64 = 8,
};

// Highest argument index this reader accepts. An argument list is short; the
// bound keeps a malformed symbol name from being used to size a vector.
inline constexpr std::uint32_t kMaxArgIndex = 4095;

// One place in the control code that takes an address.
struct PatchSite {
  std::uint32_t offset = 0; // byte offset into the control code
  std::uint32_t addend = 0; // added to the address before it is written
  PatchScheme scheme = PatchScheme::kShimDma48;
};

// One configuration image.
struct Pdi {
  std::string name; // section name, ".pdi.N"
  std::vector<std::uint8_t> data;
};

// A place in the control code that takes the device address of one PDI.
struct PdiPatch {
  PatchSite site;
  std::size_t pdi_index = 0; // index into Image::pdis
};

// A parsed kernel: the bytes to load and where addresses go.
struct Kernel {
  std::string name; // "<kernel>:<instance>", e.g. "main:q4nx_decode"
  std::vector<std::uint8_t> ctrl_code;

  // Every place the control code takes a PDI's device address.
  //
  // A design that switches configuration mid-stream has more than one: the
  // fused decode's q4nx_decode takes .pdi.2 at 0x28 and then .pdi.1 at
  // 0x569c4. The dispatch packet carries a single pdi_addr, so ROCR can fill
  // in exactly one of them -- pdi_patches.front(), reported as
  // pdi_patch_offset. The rest are written by write_control_code, which is no
  // different in kind from what the ABI already requires for arguments: they
  // are addresses of buffers this process allocated.
  std::vector<PdiPatch> pdi_patches;

  bool has_pdi_patch() const { return !pdi_patches.empty(); }
  // Offset for hsa_amd_aie_kernel_dispatch_packet_t::pdi_patch_offset.
  std::uint64_t pdi_patch_offset() const {
    return pdi_patches.empty() ? 0 : pdi_patches.front().site.offset;
  }
  // Index into Image::pdis of the PDI whose address ROCR writes, i.e. the one
  // whose buffer must be passed as the packet's pdi_addr.
  std::size_t rocr_pdi_index() const {
    return pdi_patches.empty() ? 0 : pdi_patches.front().pdi_index;
  }

  // The control scratchpad, if the design declares parameters.
  bool has_scratchpad = false;
  std::uint32_t scratchpad_size = 0; // symbol st_size, = 4 * parameter count
  std::vector<PatchSite> scratchpad_sites;

  // Patch sites per argument index. Entries may be empty for unused arguments.
  std::vector<std::vector<PatchSite>> arg_sites;

  std::uint32_t num_args() const {
    return static_cast<std::uint32_t>(arg_sites.size());
  }
};

// Everything in one full ELF: its kernels, and the PDIs they share.
//
// The PDIs are held here rather than per kernel because a file's kernels
// declare the same set (byte for byte -- the fused decode's two kernels both
// declare empty_1/empty_0/seg/main), and the largest is ~175 KiB. Holding them
// once means loading them once.
struct Image {
  std::vector<Pdi> pdis;
  std::map<std::string, Kernel> kernels;
};

namespace detail {

// Bounds-checked view over the ELF image. Returns nullptr rather than walking
// off the end, so a truncated file is a clean error.
class View {
public:
  View(const std::uint8_t *data, std::size_t size) : data_(data), size_(size) {}

  const std::uint8_t *at(std::uint64_t offset, std::uint64_t count) const {
    if (offset > size_ || count > size_ - offset)
      return nullptr;
    return data_ + offset;
  }

  template <typename T>
  const T *as(std::uint64_t offset, std::uint64_t count = 1) const {
    if (count != 0 && sizeof(T) > UINT64_MAX / count)
      return nullptr;
    return reinterpret_cast<const T *>(at(offset, sizeof(T) * count));
  }

private:
  const std::uint8_t *data_;
  std::size_t size_;
};

inline const char *string_at(const View &v, const Elf32_Shdr &strtab,
                             std::uint32_t offset) {
  if (offset >= strtab.sh_size)
    return nullptr;
  const std::uint8_t *base = v.at(strtab.sh_offset, strtab.sh_size);
  if (base == nullptr)
    return nullptr;
  const auto *str = reinterpret_cast<const char *>(base + offset);
  // The table has to contain the terminator, else the string runs off the end.
  if (std::memchr(str, '\0', strtab.sh_size - offset) == nullptr)
    return nullptr;
  return str;
}

// "_Z4mainPcPcPc" -> "main". Only the simple `_Z<len><name>` form aiecc emits
// is handled; anything else is returned unchanged, which at worst makes the
// kernel name uglier, never wrong.
inline std::string kernel_name_from_symbol(const std::string &symbol) {
  if (symbol.rfind("_Z", 0) != 0)
    return symbol;
  std::size_t i = 2;
  std::size_t len = 0;
  while (i < symbol.size() && symbol[i] >= '0' && symbol[i] <= '9') {
    len = len * 10 + static_cast<std::size_t>(symbol[i] - '0');
    ++i;
  }
  if (len == 0 || i + len > symbol.size())
    return symbol;
  return symbol.substr(i, len);
}

inline bool parse_arg_index(const char *name, std::uint32_t *index) {
  if (name == nullptr || *name == '\0')
    return false;
  std::uint32_t value = 0;
  for (const char *p = name; *p != '\0'; ++p) {
    if (*p < '0' || *p > '9')
      return false;
    const auto digit = static_cast<std::uint32_t>(*p - '0');
    // Refuse rather than wrap. A value that wrapped can land back under
    // kMaxArgIndex and name a different argument than the ELF asked for.
    if (value > (std::numeric_limits<std::uint32_t>::max() - digit) / 10)
      return false;
    value = value * 10 + digit;
  }
  *index = value;
  return true;
}

} // namespace detail

// Parse `image` into its kernels and PDIs.
//
// Throws std::runtime_error if the image is not a well-formed aie2p full ELF
// or uses a feature this reader does not implement.
inline Image parse(const std::uint8_t *image_data, std::size_t image_size) {
  const detail::View v(image_data, image_size);

  const auto *ehdr = v.as<Elf32_Ehdr>(0);
  if (ehdr == nullptr || std::memcmp(ehdr->e_ident, ELFMAG, SELFMAG) != 0 ||
      ehdr->e_ident[EI_CLASS] != ELFCLASS32 ||
      ehdr->e_ident[EI_DATA] != ELFDATA2LSB)
    throw std::runtime_error("not a little-endian ELF32");
  if (ehdr->e_ident[EI_OSABI] != kElfAmdAie2p)
    throw std::runtime_error("not an aie2p AIE ELF");
  if (ehdr->e_shentsize != sizeof(Elf32_Shdr) || ehdr->e_shnum == 0)
    throw std::runtime_error("malformed section headers");
  const std::uint8_t abi_version = ehdr->e_ident[EI_ABIVERSION];

  const auto *shdrs = v.as<Elf32_Shdr>(ehdr->e_shoff, ehdr->e_shnum);
  if (shdrs == nullptr || ehdr->e_shstrndx >= ehdr->e_shnum)
    throw std::runtime_error("malformed section headers");
  const Elf32_Shdr &shstrtab = shdrs[ehdr->e_shstrndx];

  auto section_name = [&](std::uint32_t index) -> const char * {
    if (index >= ehdr->e_shnum)
      return nullptr;
    return detail::string_at(v, shstrtab, shdrs[index].sh_name);
  };

  const Elf32_Shdr *symtab = nullptr;
  const Elf32_Shdr *strtab = nullptr;
  const Elf32_Shdr *dynsym = nullptr;
  const Elf32_Shdr *dynstr = nullptr;
  const Elf32_Shdr *rela = nullptr;
  for (std::uint32_t i = 0; i < ehdr->e_shnum; ++i) {
    const char *name = section_name(i);
    if (name == nullptr)
      continue;
    if (std::strcmp(name, ".symtab") == 0)
      symtab = &shdrs[i];
    else if (std::strcmp(name, ".strtab") == 0)
      strtab = &shdrs[i];
    else if (std::strcmp(name, ".dynsym") == 0)
      dynsym = &shdrs[i];
    else if (std::strcmp(name, ".dynstr") == 0)
      dynstr = &shdrs[i];
    else if (std::strcmp(name, ".rela.dyn") == 0)
      rela = &shdrs[i];
  }
  if (symtab == nullptr || strtab == nullptr ||
      symtab->sh_entsize != sizeof(Elf32_Sym))
    throw std::runtime_error("missing or malformed .symtab");

  const std::uint32_t symtab_count = symtab->sh_size / sizeof(Elf32_Sym);
  const auto *symbols = v.as<Elf32_Sym>(symtab->sh_offset, symtab_count);
  if (symbols == nullptr)
    throw std::runtime_error("malformed .symtab");

  auto section_bytes = [&](std::uint32_t index) {
    const std::uint8_t *data =
        v.at(shdrs[index].sh_offset, shdrs[index].sh_size);
    if (data == nullptr)
      throw std::runtime_error("section extends past end of file");
    return std::vector<std::uint8_t>(data, data + shdrs[index].sh_size);
  };

  Image out;

  // ---- PDIs ---------------------------------------------------------------
  // Collected by section name prefix rather than through the groups, because
  // the groups hold only the control code. A file with several kernels repeats
  // each PDI once per kernel under the same name and with identical bytes
  // (verified on the fused decode: .pdi.1..4 appear twice, byte for byte), so
  // deduplicate by name -- otherwise the decode's 175 KiB seg.pdi is loaded
  // twice for no reason. A name that recurs with *different* bytes would make
  // the relocation's symbol name ambiguous, so refuse it rather than pick one.
  std::map<std::string, std::size_t> pdi_by_name;
  for (std::uint32_t i = 0; i < ehdr->e_shnum; ++i) {
    const char *name = section_name(i);
    if (name == nullptr || std::strncmp(name, ".pdi", 4) != 0)
      continue;
    if (shdrs[i].sh_type != SHT_PROGBITS)
      continue;
    auto bytes = section_bytes(i);
    auto it = pdi_by_name.find(name);
    if (it != pdi_by_name.end()) {
      if (out.pdis[it->second].data != bytes)
        throw std::runtime_error(
            std::string("two sections named '") + name +
            "' hold different bytes, so a relocation naming it is ambiguous");
      continue;
    }
    pdi_by_name.emplace(name, out.pdis.size());
    out.pdis.push_back(Pdi{name, std::move(bytes)});
  }

  // ---- groups -> kernels --------------------------------------------------
  // A group's sh_info is the .symtab index of its instance symbol, whose
  // st_shndx is in turn the .symtab index of the kernel's function symbol.
  // That is an overload of st_shndx specific to this ELF flavour, not a
  // section index.
  struct Group {
    std::string name;
    std::uint32_t ctrltext_section = 0;
  };
  std::map<std::uint32_t, Group> groups; // group section index -> group
  std::map<std::uint32_t, std::uint32_t> sec2grp; // member section -> group

  for (std::uint32_t i = 0; i < ehdr->e_shnum; ++i) {
    if (shdrs[i].sh_type != SHT_GROUP)
      continue;
    if (shdrs[i].sh_info >= symtab_count)
      throw std::runtime_error("bad group signature symbol");

    const Elf32_Sym &instance_sym = symbols[shdrs[i].sh_info];
    const char *instance_name =
        detail::string_at(v, *strtab, instance_sym.st_name);
    if (instance_name == nullptr || instance_sym.st_shndx >= symtab_count)
      throw std::runtime_error("bad group signature symbol");
    const char *kernel_sym =
        detail::string_at(v, *strtab, symbols[instance_sym.st_shndx].st_name);
    if (kernel_sym == nullptr)
      throw std::runtime_error("bad kernel symbol");

    Group group;
    group.name =
        detail::kernel_name_from_symbol(kernel_sym) + ":" + instance_name;

    // Group data is a flags word followed by the member section indices.
    const std::uint32_t word_count = shdrs[i].sh_size / sizeof(Elf32_Word);
    const auto *words = v.as<Elf32_Word>(shdrs[i].sh_offset, word_count);
    if (words == nullptr)
      throw std::runtime_error("malformed group section");
    for (std::uint32_t w = 1; w < word_count; ++w) {
      const std::uint32_t member = words[w];
      if (member >= ehdr->e_shnum)
        throw std::runtime_error("group member out of range");
      sec2grp[member] = i;
      const char *member_name = section_name(member);
      if (member_name != nullptr &&
          std::strncmp(member_name, ".ctrltext", 9) == 0) {
        if (shdrs[member].sh_type != SHT_PROGBITS)
          throw std::runtime_error("control code section holds no data");
        group.ctrltext_section = member;
      }
    }
    groups.emplace(i, std::move(group));
  }
  if (groups.empty())
    throw std::runtime_error("no COMDAT groups: not a group ELF");

  // ---- relocations --------------------------------------------------------
  // Each names a symbol whose st_shndx is the section being patched and whose
  // name says what address to write: ".pdi.N" for a PDI, "scratch-pad-ctrl"
  // for the scratchpad, a decimal string for an argument.
  struct KernelSites {
    std::map<std::uint32_t, std::vector<PatchSite>> args;
    std::vector<PatchSite> scratchpad;
    std::uint32_t scratchpad_size = 0;
    bool has_scratchpad = false;
    std::vector<PdiPatch> pdi_patches;
  };
  std::map<std::uint32_t, KernelSites> sites;

  if (rela != nullptr && dynsym != nullptr && dynstr != nullptr) {
    if (rela->sh_entsize != sizeof(Elf32_Rela) ||
        dynsym->sh_entsize != sizeof(Elf32_Sym))
      throw std::runtime_error("malformed .rela.dyn");
    const std::uint32_t rela_count = rela->sh_size / sizeof(Elf32_Rela);
    const auto *relocs = v.as<Elf32_Rela>(rela->sh_offset, rela_count);
    const std::uint32_t dynsym_count = dynsym->sh_size / sizeof(Elf32_Sym);
    const auto *dynsyms = v.as<Elf32_Sym>(dynsym->sh_offset, dynsym_count);
    if (relocs == nullptr || dynsyms == nullptr)
      throw std::runtime_error("malformed .rela.dyn");

    for (std::uint32_t r = 0; r < rela_count; ++r) {
      const std::uint32_t sym_index = ELF32_R_SYM(relocs[r].r_info);
      if (sym_index >= dynsym_count)
        throw std::runtime_error("relocation symbol out of range");
      const Elf32_Sym &sym = dynsyms[sym_index];
      const char *sym_name = detail::string_at(v, *dynstr, sym.st_name);
      if (sym_name == nullptr)
        throw std::runtime_error("bad relocation symbol name");

      auto grp_it = sec2grp.find(sym.st_shndx);
      if (grp_it == sec2grp.end())
        continue;
      const Group &group = groups.at(grp_it->second);
      if (sym.st_shndx != group.ctrltext_section)
        continue; // only control code is patched

      PatchSite site;
      site.offset = relocs[r].r_offset;
      if (abi_version == 1) {
        // In ABI version 1 the scheme lives in the low bits of the addend
        // rather than in r_info.
        site.addend = static_cast<std::uint32_t>(relocs[r].r_addend) >> 4;
        site.scheme = static_cast<PatchScheme>(relocs[r].r_addend & 0xF);
      } else {
        site.addend = static_cast<std::uint32_t>(relocs[r].r_addend);
        site.scheme = static_cast<PatchScheme>(ELF32_R_TYPE(relocs[r].r_info));
      }

      KernelSites &ks = sites[grp_it->second];

      if (std::strcmp(sym_name, kScratchpadSymbol) == 0) {
        if (site.scheme != PatchScheme::kAddress64)
          throw std::runtime_error(
              "unexpected patch scheme for the scratchpad");
        // st_size is the buffer size: 4 bytes per declared parameter, the same
        // number params.txt counts.
        if (ks.has_scratchpad && ks.scratchpad_size != sym.st_size)
          throw std::runtime_error("scratchpad declared with two sizes");
        ks.has_scratchpad = true;
        ks.scratchpad_size = sym.st_size;
        ks.scratchpad.push_back(site);
        continue;
      }

      if (std::strncmp(sym_name, ".pdi", 4) == 0) {
        if (site.scheme != PatchScheme::kAddress64)
          throw std::runtime_error("unexpected patch scheme for PDI symbol");
        auto pdi_it = pdi_by_name.find(sym_name);
        if (pdi_it == pdi_by_name.end())
          throw std::runtime_error(std::string("PDI section not found: ") +
                                   sym_name);
        ks.pdi_patches.push_back(PdiPatch{site, pdi_it->second});
        continue;
      }

      std::uint32_t arg_index = 0;
      if (!detail::parse_arg_index(sym_name, &arg_index))
        throw std::runtime_error(
            std::string("unsupported relocation symbol: ") + sym_name);
      if (site.scheme != PatchScheme::kShimDma48)
        throw std::runtime_error(
            "unsupported patch scheme for a kernel argument");
      if (arg_index > kMaxArgIndex)
        throw std::runtime_error("kernel argument index out of range");
      ks.args[arg_index].push_back(site);
    }
  }

  // ---- assemble -----------------------------------------------------------
  for (auto &[grp_index, group] : groups) {
    if (group.ctrltext_section == 0)
      continue; // nothing to dispatch

    Kernel k;
    k.name = group.name;
    k.ctrl_code = section_bytes(group.ctrltext_section);
    if (k.ctrl_code.empty())
      throw std::runtime_error("empty control code");

    auto it = sites.find(grp_index);
    if (it != sites.end()) {
      KernelSites &ks = it->second;

      // Lowest offset first, so pdi_patches.front() -- the one ROCR fills in --
      // is the configuration the stream loads before it runs anything, not
      // whichever relocation happened to come first in .rela.dyn.
      k.pdi_patches = std::move(ks.pdi_patches);
      std::sort(k.pdi_patches.begin(), k.pdi_patches.end(),
                [](const PdiPatch &a, const PdiPatch &b) {
                  return a.site.offset < b.site.offset;
                });
      for (const PdiPatch &p : k.pdi_patches) {
        // A 64-bit address is written here, so it has to lie wholly inside the
        // control code.
        if (p.site.offset + sizeof(std::uint64_t) > k.ctrl_code.size() ||
            p.site.offset % sizeof(std::uint32_t) != 0)
          throw std::runtime_error(
              "PDI patch site does not fit the control code");
      }
      // Offset 0 is how pdi_patch_offset spells "PDI plus instruction
      // sequence", so a kernel reporting it would silently take the other
      // dispatch shape. A real full-ELF control code opens with a transaction
      // header and never puts the patch site there.
      if (k.has_pdi_patch() && k.pdi_patch_offset() == 0)
        throw std::runtime_error(
            "PDI patch site at offset 0 is indistinguishable from no patch");

      if (ks.has_scratchpad) {
        k.has_scratchpad = true;
        k.scratchpad_size = ks.scratchpad_size;
        k.scratchpad_sites = std::move(ks.scratchpad);
        if (k.scratchpad_size == 0 || k.scratchpad_size % sizeof(std::uint32_t))
          throw std::runtime_error("scratchpad size " +
                                   std::to_string(k.scratchpad_size) +
                                   " is not a positive multiple of 4");
        for (const PatchSite &s : k.scratchpad_sites)
          if (s.offset + sizeof(std::uint64_t) > k.ctrl_code.size() ||
              s.offset % sizeof(std::uint32_t) != 0)
            throw std::runtime_error(
                "scratchpad patch site does not fit the control code");
      }

      if (!ks.args.empty()) {
        const std::uint32_t max_arg = ks.args.rbegin()->first;
        k.arg_sites.resize(max_arg + 1);
        for (auto &[arg_index, s] : ks.args)
          k.arg_sites[arg_index] = std::move(s);
      }
    }

    out.kernels.emplace(k.name, std::move(k));
  }
  if (out.kernels.empty())
    throw std::runtime_error("no dispatchable kernels");
  return out;
}

// Read an ELF from disk and parse it.
inline Image parse_file(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f)
    throw std::runtime_error("cannot open " + path);
  const auto size = static_cast<std::size_t>(f.tellg());
  f.seekg(0);
  std::vector<std::uint8_t> bytes(size);
  f.read(reinterpret_cast<char *>(bytes.data()),
         static_cast<std::streamsize>(size));
  if (static_cast<std::size_t>(f.gcount()) != size)
    throw std::runtime_error("short read " + path);
  return parse(bytes.data(), bytes.size());
}

// Fold a buffer address into a shim DMA buffer descriptor, the scheme the NPU
// firmware defines. This *adds* to the descriptor already in place, so it must
// only ever be applied to a pristine copy of the control code -- see
// write_control_code.
inline void patch_shim_dma48(std::uint32_t *site, std::uint64_t addr) {
  constexpr std::uint64_t kDdrAieAddrOffset = 0x80000000;
  std::uint64_t base = ((static_cast<std::uint64_t>(site[2]) & 0xFFFF) << 32) |
                       static_cast<std::uint64_t>(site[1]);
  base += addr + kDdrAieAddrOffset;
  site[1] = static_cast<std::uint32_t>(base & 0xFFFFFFFC);
  site[2] = (site[2] & 0xFFFF0000) | static_cast<std::uint32_t>(base >> 32);
}

// Store a plain 64-bit little-endian address.
inline void patch_address64(std::uint32_t *site, std::uint64_t addr) {
  site[0] = static_cast<std::uint32_t>(addr & 0xFFFFFFFFu);
  site[1] = static_cast<std::uint32_t>(addr >> 32);
}

// Write a dispatch-ready copy of `kernel`'s control code into `dst`, with the
// arguments, the scratchpad and every PDI address but ROCR's patched in.
//
// `dst` must be at least kernel.ctrl_code.size() bytes and allocated from the
// agent's device memory pool, 16 KiB aligned. Always writes the pristine
// control code first: the shim DMA scheme is additive, so patching over a
// previous result would accumulate.
//
// `pdi_addrs` is indexed by Image::pdis position. The *first* PDI patch site is
// deliberately skipped -- ROCR writes that one at pdi_patch_offset() when the
// packet is submitted, from the packet's pdi_addr. Writing it here too would be
// harmless but redundant; leaving it to ROCR is what makes the dispatch
// full-ELF in the first place.
inline void write_control_code(const Kernel &kernel, void *dst,
                               std::size_t dst_size,
                               const std::vector<std::uint64_t> &arg_addrs,
                               std::uint64_t scratchpad_addr,
                               const std::vector<std::uint64_t> &pdi_addrs) {
  if (dst_size < kernel.ctrl_code.size())
    throw std::runtime_error("control code buffer too small");
  if (arg_addrs.size() < kernel.num_args())
    throw std::runtime_error("too few argument addresses");
  if (kernel.has_scratchpad && scratchpad_addr == 0)
    throw std::runtime_error(
        "kernel declares a scratchpad but no scratchpad address was given");

  auto *out = static_cast<std::uint8_t *>(dst);
  std::memcpy(out, kernel.ctrl_code.data(), kernel.ctrl_code.size());

  auto site_ptr = [&](const PatchSite &s, std::size_t words) {
    if (s.offset % sizeof(std::uint32_t) != 0 ||
        s.offset + words * sizeof(std::uint32_t) > kernel.ctrl_code.size())
      throw std::runtime_error("patch site out of range");
    return reinterpret_cast<std::uint32_t *>(out + s.offset);
  };

  for (std::uint32_t arg = 0; arg < kernel.arg_sites.size(); ++arg)
    for (const PatchSite &s : kernel.arg_sites[arg])
      // The scheme reads and writes three dwords from the patch site.
      patch_shim_dma48(site_ptr(s, 3), arg_addrs[arg] + s.addend);

  for (const PatchSite &s : kernel.scratchpad_sites)
    patch_address64(site_ptr(s, 2), scratchpad_addr + s.addend);

  // Skip index 0: that is ROCR's, written from the packet's pdi_addr.
  for (std::size_t i = 1; i < kernel.pdi_patches.size(); ++i) {
    const PdiPatch &p = kernel.pdi_patches[i];
    if (p.pdi_index >= pdi_addrs.size())
      throw std::runtime_error("no address given for a PDI the control code "
                               "switches to");
    patch_address64(site_ptr(p.site, 2),
                    pdi_addrs[p.pdi_index] + p.site.addend);
  }
}

} // namespace aie_full_elf
