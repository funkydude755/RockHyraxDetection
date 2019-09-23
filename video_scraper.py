import csv
import googleapiclient.discovery
import googleapiclient
from pytube import YouTube


def rock_hyrax_videos(yt, **kwargs):
    print(".")
    video_list = yt.search().list(q='rock hyrax', part="id", maxResults=50, type="video").execute()
    for video in video_list['items']:
        # print(video['id'])
        # ls =
        yield video['id']

yt = googleapiclient.discovery.build('youtube', 'v3')

for i in rock_hyrax_videos(yt):
    print (i)
    YouTube(video_id=i['videoId']).streams.first().download('rock_hyrax_videos/')
