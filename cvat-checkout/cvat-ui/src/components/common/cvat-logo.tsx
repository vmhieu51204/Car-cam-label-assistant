// Copyright (C) CVAT.ai Corporation
// SPDX-License-Identifier: MIT

import React from 'react';
import logo from '../../assets/ai-x-logo.jpg';
import './ai-x-logo.scss';

function CVATLogo(): JSX.Element {
    return (
        <div className='cvat-logo-icon ai-x-logo'>
            <svg viewBox='452 165 826 278' preserveAspectRatio='xMidYMid meet' role='img' aria-label='AI-X'>
                <title>AI-X</title>
                <image href={logo} x='0' y='0' width='1725' height='608' />
            </svg>
        </div>
    );
}

export default React.memo(CVATLogo);
