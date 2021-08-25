function z = audio_realtime2()
% init recording parameters
fs = 44100;
period = 44100;
nbmic = 2;
ampMax = 0.05;
ampMin = -0.05;

% declare sound card
audioIn = audioDeviceReader(...
'Device', 'Microphone Array(AMD Audio Device)',...
'Driver', 'DirectSound', ...
'SampleRate', fs, ...
'NumChannels', nbmic ,...
'OutputDataType','double',...
'SamplesPerFrame', period,...
'BitDepth','24-bit integer');

audioBuffer = dsp.AsyncBuffer(fs);

h = figure('Units','normalized','Position',[0.2 0.1 0.6 0.8]);

timeLimit = 20;

tic
while ishandle(h) && toc < timeLimit

    % Extract audio samples from the audio device and add the samples to
    % the buffer.
    x = audioIn();
    write(audioBuffer,x);
    y = read(audioBuffer,fs,fs-audioIn.SamplesPerFrame);
    % Plot the current waveform and spectrogram.
    subplot(2,1,1)
    plot(y)
    axis tight
    ylim([-0.4,0.4])
	xlabel('Time')
	ylabel('Amplitude')
    drawnow
end



nfft = 2^12;
