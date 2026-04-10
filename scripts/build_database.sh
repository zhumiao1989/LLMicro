#!/bin/bash
#
# Build Reference Databases for LLMicro
#
# This script builds reference databases for Kraken2, Centrifuge, and PathSeq
# from a unified set of reference sequences.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration
DATA_DIR="$PROJECT_DIR/data"
REFERENCE_DIR="$DATA_DIR/reference"
TEMP_DIR="$PROJECT_DIR/temp"

# Reference sequence sources (GenBank 202404)
# In practice, these would be downloaded from NCBI GenBank
BACTERIA_DIR="$REFERENCE_DIR/raw/bacteria"
FUNGI_DIR="$REFERENCE_DIR/raw/fungi"
PARASITES_DIR="$REFERENCE_DIR/raw/parasites"
VIRUSES_DIR="$REFERENCE_DIR/raw/viruses"
HOST_DIR="$REFERENCE_DIR/raw/host"

# Output directories
KRAKEN2_DB="$REFERENCE_DIR/kraken2"
CENTRIFUGE_DB="$REFERENCE_DIR/centrifuge"
PATHSEQ_DB="$REFERENCE_DIR/pathseq"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Create directories
log_info "Creating directories..."
mkdir -p "$REFERENCE_DIR"/{raw,kraken2,centrifuge,pathseq}
mkdir -p "$TEMP_DIR"

# Step 1: Download reference sequences (placeholder)
log_info "Step 1: Reference sequence download (placeholder)"
log_warn "This step requires actual sequence downloads from GenBank"
log_warn "For testing, create placeholder sequences..."

# Create placeholder sequences for testing
mkdir -p "$BACTERIA_DIR" "$FUNGI_DIR" "$VIRUSES_DIR" "$HOST_DIR"

# Generate a small test genome for each major group
cat > "$BACTERIA_DIR/ecoli.fasta" << 'EOF'
>NC_000913.3 Escherichia coli str. K-12 substr. MG1655
ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG
ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGAT
CGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGA
EOF

cat > "$FUNGI_DIR/yeast.fasta" << 'EOF'
>NC_001148.6 Saccharomyces cerevisiae chromosome I
ATGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG
CTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG
CTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG
EOF

cat > "$VIRUSES_DIR/phix.fasta" << 'EOF'
>NC_001422.1 Enterobacteria phage phiX174
ATGPHIX174BACTERIOPHAGEGENOMESEQUENCEPHIX174BACTERIOPHAGEGENOMESEQUENCE
PHIX174BACTERIOPHAGEGENOMESEQUENCEPHIX174BACTERIOPHAGEGENOMESEQUENCEPHIX
EOF

cat > "$HOST_DIR/hg38.fasta" << 'EOF'
>hg38 Human reference genome (placeholder)
ATGHG38HUMANREFERENCEGENOMESEQUENCEHG38HUMANREFERENCEGENOMESEQUENCEHG38
HUMANREFERENCEGENOMESEQUENCEHG38HUMANREFERENCEGENOMESEQUENCEHG38HUMANREF
EOF

log_info "Placeholder sequences created"

# Step 2: Build Kraken2 database
log_info "Step 2: Building Kraken2 database..."

if command -v kraken2-build &> /dev/null; then
    # Create Kraken2 database
    kraken2-build --initialize --db "$KRAKEN2_DB"
    kraken2-build --download-taxonomy --db "$KRAKEN2_DB"

    # Add sequences from each group
    for fasta in "$BACTERIA_DIR"/*.fasta "$FUNGI_DIR"/*.fasta "$VIRUSES_DIR"/*.fasta; do
        if [ -f "$fasta" ]; then
            kraken2-build --add-to-library "$fasta" --db "$KRAKEN2_DB"
        fi
    done

    kraken2-build --build --db "$KRAKEN2_DB"
    log_info "Kraken2 database built successfully"
else
    log_warn "kraken2-build not found. Skipping Kraken2 database build."
fi

# Step 3: Build Centrifuge database
log_info "Step 3: Building Centrifuge database..."

if command -v centrifuge-build &> /dev/null; then
    # Create sequence list
    SEQUENCE_LIST="$TEMP_DIR/centrifuge_sequences.txt"
    > "$SEQUENCE_LIST"

    for fasta in "$BACTERIA_DIR"/*.fasta "$FUNGI_DIR"/*.fasta "$VIRUSES_DIR"/*.fasta; do
        if [ -f "$fasta" ]; then
            echo "$fasta" >> "$SEQUENCE_LIST"
        fi
    done

    # Build database
    centrifuge-build -p 8 \
        --taxonomy-file "$REFERENCE_DIR/raw/taxonomy/nodes.dmp" \
        --name-table "$REFERENCE_DIR/raw/taxonomy/names.dmp" \
        --input-file "$SEQUENCE_LIST" \
        --output-file "$CENTRIFUGE_DB/centrifuge" \
        --conversion-table "$CENTRIFUGE_DB/conversion_table.txt"

    log_info "Centrifuge database built successfully"
else
    log_warn "centrifuge-build not found. Skipping Centrifuge database build."
fi

# Step 4: Build PathSeq database
log_info "Step 4: Building PathSeq database..."

if command -v gatk &> /dev/null; then
    # Concatenate all microbial sequences
    MICROBE_FASTA="$TEMP_DIR/microbe_combined.fasta"
    cat "$BACTERIA_DIR"/*.fasta "$FUNGI_DIR"/*.fasta "$VIRUSES_DIR"/*.fasta > "$MICROBE_FASTA"

    # Build kmer files
    gatk PathSeqPreprocessKmers \
        --microbe-reference "$MICROBE_FASTA" \
        --microbe-kmer-file "$PATHSEQ_DB/microbe.kmer" \
        --host-kmer-file "$PATHSEQ_DB/host.kmer" \
        --host-reference "$HOST_DIR/hg38.fasta"

    log_info "PathSeq database built successfully"
else
    log_warn "gatk not found. Skipping PathSeq database build."
fi

# Cleanup
log_info "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

log_info "Database build complete!"
log_info ""
log_info "Database locations:"
log_info "  Kraken2: $KRAKEN2_DB"
log_info "  Centrifuge: $CENTRIFUGE_DB"
log_info "  PathSeq: $PATHSEQ_DB"
