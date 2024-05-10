Name:       nnstreamer-whisper-cpp
Summary:    nnstreamer-whisper-cpp shared library
Version:    1.0.0
Release:    0
Group:      Machine Learning/ML Framework
Packager:   Yongjoo Ahn <yongjoo1.ahn@samsung.com>
License:    LGPL-2.1
Source0:    %{name}-%{version}.tar.gz
Source1001: %{name}.manifest
Source2001: ggml-tiny.en.bin
Source2002: gb0.wav

Requires:	nnstreamer

%ifarch armv7l
BuildRequires: clang-accel-armv7l-cross-arm
%endif

%ifarch armv7hl
BuildRequires: clang-accel-armv7hl-cross-arm
%endif

%ifarch aarch64
BuildRequires: clang-accel-aarch64-cross-aarch64
%endif

BuildRequires:  clang
BuildRequires:  pkg-config
BuildRequires:  nnstreamer-devel

%define     nnstexampledir	/usr/lib/nnstreamer/bin

%description
nnstreamer-whisper-cpp shared library

%prep
%setup -q
cp %{SOURCE1001} .
cp %{SOURCE2001} models/
cp %{SOURCE2002} samples/

%build

CLANG_ASMFLAGS=" --target=%{_host} "
CLANG_CFLAGS=" --target=%{_host} "
CLANG_CXXFLAGS=" --target=%{_host} "

export ASMFLAGS="${CLANG_ASMFLAGS}"
export CFLAGS="${CLANG_CFLAGS}"
export CXXFLAGS="${CLANG_CXXFLAGS}"

export CC=clang
export CXX=clang++

make nnstreamer

%install
mkdir -p %{buildroot}%{nnstexampledir}/models
cp libnnstreamer-whisper.so %{buildroot}%{nnstexampledir}
cp models/ggml-tiny.en.bin %{buildroot}%{nnstexampledir}/models
cp samples/gb0.wav %{buildroot}%{nnstexampledir}

%files
%manifest nnstreamer-whisper-cpp.manifest
%defattr(-,root,root,-)
%{nnstexampledir}/*

%changelog
* Fri May 10 2024 Yongjoo Ahn <yongjoo1.ahn@samsung.com>
- create the nnstreamer-whisper-cpp shared library
