import os
import wget

classifier_urls = [
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-00-male.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-01-smiling.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-02-attractive.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-03-wavy-hair.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-04-young.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-05-5-o-clock-shadow.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-06-arched-eyebrows.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-07-bags-under-eyes.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-08-bald.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-09-bangs.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-10-big-lips.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-11-big-nose.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-12-black-hair.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-13-blond-hair.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-14-blurry.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-15-brown-hair.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-16-bushy-eyebrows.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-17-chubby.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-18-double-chin.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-19-eyeglasses.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-20-goatee.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-21-gray-hair.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-22-heavy-makeup.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-23-high-cheekbones.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-24-mouth-slightly-open.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-25-mustache.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-26-narrow-eyes.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-27-no-beard.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-28-oval-face.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-29-pale-skin.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-30-pointy-nose.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-31-receding-hairline.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-32-rosy-cheeks.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-33-sideburns.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-34-straight-hair.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-35-wearing-earrings.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-36-wearing-hat.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-37-wearing-lipstick.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-38-wearing-necklace.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-39-wearing-necktie.pkl',
]

selected_attrs = [
    "male",          # 0
    "smiling",       # 1
    "young",         # 2
    "eyeglasses"     # 3
]

for url in classifier_urls:
    for attr in selected_attrs:
        if attr in url:
            wget.download(url, "./")

