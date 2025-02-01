// Copyright (c) 2009-2010 Satoshi Nakamoto
// Copyright (c) 2009-2019 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <primitives/block.h>
#include <crypto/tens_pow/tens_hash.h>

#include <hash.h>
#include <tinyformat.h>

uint256 CBlockHeader::GetHash() const
{
    // Create a copy of the header
    CBlockHeader header = *this;
    // Zero out the nonce
    header.nNonce = 0;
    // Hash the modified header
    return (HashWriter{} << header).GetHash();
}

uint256 CBlockHeader::GetPoWHash() const {
    uint256 thash;
    uint256 seed=GetHash();
    uint8_t nonce_bytes[32] = {0};
    memcpy(nonce_bytes, &nNonce, sizeof(nNonce));
    tens_hash(nonce_bytes, seed.begin(), thash.begin());
    return thash;
}

std::string CBlock::ToString() const
{
    std::stringstream s;
    s << strprintf("CBlock(hash=%s, ver=0x%08x, hashPrevBlock=%s, hashMerkleRoot=%s, nTime=%u, nBits=%08x, nNonce=%u, vtx=%u)\n",
        GetHash().ToString(),
        nVersion,
        hashPrevBlock.ToString(),
        hashMerkleRoot.ToString(),
        nTime, nBits, nNonce,
        vtx.size());
    for (const auto& tx : vtx) {
        s << "  " << tx->ToString() << "\n";
    }
    return s.str();
}
